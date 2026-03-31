"""訊號歷史資料庫模組"""
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = Path.home() / ".openclaw" / "workspace" / "crypto-agent-platform" / "data" / "signals.db"


def get_db_path() -> Path:
    """取得資料庫路徑"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DB_PATH


@contextmanager
def get_connection():
    """取得資料庫連線（上下文管理器）"""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """初始化資料庫"""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                direction TEXT NOT NULL,
                price REAL NOT NULL,
                strategy TEXT NOT NULL,
                ai_confidence REAL,
                ai_rating TEXT,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 建立索引加速查詢
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp 
            ON signals(symbol, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_timestamp 
            ON signals(timestamp)
        """)
        
        logger.info(f"Database initialized at {get_db_path()}")


def save_signal(
    symbol: str,
    signal: str,
    direction: str,
    price: float,
    strategy: str,
    ai_confidence: Optional[float] = None,
    ai_rating: Optional[str] = None,
    timestamp: Optional[str] = None
) -> int:
    """儲存訊號到資料庫"""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signals 
            (symbol, signal, direction, price, strategy, ai_confidence, ai_rating, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, signal, direction, price, strategy, ai_confidence, ai_rating, timestamp))
        
        signal_id = cursor.lastrowid
        logger.debug(f"Signal saved: id={signal_id}, {symbol} {direction} @ {price}")
        return signal_id


def get_signal_history(
    symbol: Optional[str] = None,
    days: int = 7,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """取得歷史訊號"""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute("""
                SELECT * FROM signals 
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (symbol, cutoff, limit))
        else:
            cursor.execute("""
                SELECT * FROM signals 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (cutoff, limit))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_signal_stats(symbol: str, days: int = 30) -> Dict[str, Any]:
    """取得訊號統計"""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 基本統計
        cursor.execute("""
            SELECT 
                COUNT(*) as total_signals,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT strategy) as unique_strategies,
                AVG(ai_confidence) as avg_confidence
            FROM signals 
            WHERE symbol = ? AND timestamp >= ?
        """, (symbol, cutoff))
        
        row = cursor.fetchone()
        stats = dict(row) if row else {}
        
        # 按方向統計
        cursor.execute("""
            SELECT direction, COUNT(*) as count 
            FROM signals 
            WHERE symbol = ? AND timestamp >= ?
            GROUP BY direction
        """, (symbol, cutoff))
        
        direction_counts = {row["direction"]: row["count"] for row in cursor.fetchall()}
        stats["direction_counts"] = direction_counts
        
        # 按策略統計
        cursor.execute("""
            SELECT strategy, COUNT(*) as count, AVG(ai_confidence) as avg_conf
            FROM signals 
            WHERE symbol = ? AND timestamp >= ?
            GROUP BY strategy
        """, (symbol, cutoff))
        
        stats["strategy_stats"] = [dict(row) for row in cursor.fetchall()]
        
        # 計算勝率（假設 direction=LONG 且後續價格上漲為勝）
        # 這需要對比後續價格，但目前我們用更簡單的方式：統計各方向訊號數
        total_long = direction_counts.get("LONG", 0)
        total_short = direction_counts.get("SHORT", 0)
        total = total_long + total_short
        
        if total > 0:
            stats["long_ratio"] = total_long / total
            stats["short_ratio"] = total_short / total
        
        # 訊號頻率（每天平均）
        stats["avg_signals_per_day"] = total / days if days > 0 else 0
        
        return stats


def get_recent_signals_with_outcome(symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """取得最近訊號（帶後續走勢分析）"""
    signals = get_signal_history(symbol=symbol, days=30, limit=limit)
    
    # 對每個訊號計算後續價格變化
    from binance_client import binance_client
    
    result = []
    for sig in signals:
        try:
            # 取得訊號時間的價格
            signal_time = datetime.fromisoformat(sig["timestamp"])
            sig_price = sig["price"]
            
            # 嘗試取得最新價格計算變化
            try:
                current_price = binance_client.get_symbol_price(sig["symbol"])
                if current_price:
                    price_change = ((current_price - sig_price) / sig_price) * 100
                    sig["current_price"] = current_price
                    sig["price_change_pct"] = round(price_change, 2)
                    
                    # 簡單判定贏虧
                    if sig["direction"] == "LONG":
                        sig["outcome"] = "WIN" if price_change > 0 else "LOSS"
                    else:
                        sig["outcome"] = "WIN" if price_change < 0 else "LOSS"
                else:
                    sig["outcome"] = "PENDING"
            except:
                sig["outcome"] = "PENDING"
                
        except Exception as e:
            logger.debug(f"Could not calculate outcome for signal {sig.get('id')}: {e}")
            sig["outcome"] = "PENDING"
        
        result.append(sig)
    
    return result


# 初始化資料庫
init_db()
