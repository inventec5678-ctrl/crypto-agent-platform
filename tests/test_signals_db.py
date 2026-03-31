"""
訊號資料庫單元測試
"""
import pytest
import sys
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSignalsDb:
    """測試訊號資料庫功能"""

    def setup_method(self):
        """設置測試環境 - 使用臨時資料庫"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name

    def teardown_method(self):
        """清理測試資料庫"""
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_init_db_creates_table(self):
        """測試初始化資料庫"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.db_path):
            signals_db.init_db()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
            result = cursor.fetchone()
            conn.close()

            assert result is not None
            assert result[0] == "signals"

    def test_init_db_creates_indexes(self):
        """測試初始化時建立索引"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.db_path):
            signals_db.init_db()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            conn.close()

            assert any("symbol_timestamp" in idx for idx in indexes)

    def test_save_signal_returns_id(self):
        """測試儲存訊號並回傳 ID"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.db_path):
            signals_db.init_db()
            signal_id = signals_db.save_signal(
                symbol="BTCUSDT",
                signal="BUY",
                direction="LONG",
                price=50000.0,
                strategy="MA_Cross",
                ai_confidence=75.0,
                ai_rating="★★",
            )

            assert signal_id is not None
            assert signal_id >= 1

    def test_save_and_get_signal_roundtrip(self):
        """測試儲存後可正確取出"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.db_path):
            signals_db.init_db()

            ts = datetime.now().isoformat()
            sig_id = signals_db.save_signal(
                symbol="ETHUSDT",
                signal="SELL",
                direction="SHORT",
                price=3000.0,
                strategy="RSI_Reversal",
                ai_confidence=65.0,
                ai_rating="★★",
                timestamp=ts,
            )

            history = signals_db.get_signal_history(symbol="ETHUSDT", limit=10)
            assert len(history) == 1
            assert history[0]["symbol"] == "ETHUSDT"
            assert history[0]["signal"] == "SELL"
            assert history[0]["price"] == 3000.0

    def test_get_signal_history_respects_limit(self):
        """測試 get_signal_history 限制數量"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.db_path):
            signals_db.init_db()

            for i in range(20):
                signals_db.save_signal(
                    symbol="BTCUSDT",
                    signal="BUY",
                    direction="LONG",
                    price=50000 + i,
                    strategy="Test",
                    ai_confidence=70.0,
                    ai_rating="★★",
                    timestamp=(datetime.now() - timedelta(hours=i)).isoformat(),
                )

            history = signals_db.get_signal_history(limit=5)
            assert len(history) == 5

    def test_get_signal_history_filters_by_days(self):
        """測試以天數篩選"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.db_path):
            signals_db.init_db()

            # 舊的訊號（10 天前）
            old_ts = (datetime.now() - timedelta(days=10)).isoformat()
            signals_db.save_signal(
                symbol="BTCUSDT", signal="BUY", direction="LONG",
                price=49000, strategy="Old",
                timestamp=old_ts,
            )

            # 新的訊號（1 天前）
            new_ts = (datetime.now() - timedelta(days=1)).isoformat()
            signals_db.save_signal(
                symbol="BTCUSDT", signal="SELL", direction="SHORT",
                price=51000, strategy="New",
                timestamp=new_ts,
            )

            # 只取 7 天內
            history = signals_db.get_signal_history(days=7)
            assert len(history) == 1
            assert history[0]["strategy"] == "New"

    def test_get_signal_stats(self):
        """測試訊號統計"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.db_path):
            signals_db.init_db()

            for _ in range(3):
                signals_db.save_signal(
                    symbol="BTCUSDT", signal="BUY", direction="LONG",
                    price=50000, strategy="MA_Cross",
                    ai_confidence=75.0,
                )

            for _ in range(2):
                signals_db.save_signal(
                    symbol="BTCUSDT", signal="SELL", direction="SHORT",
                    price=50000, strategy="RSI_Reversal",
                    ai_confidence=65.0,
                )

            stats = signals_db.get_signal_stats(symbol="BTCUSDT", days=30)
            assert stats["total_signals"] == 5
            assert stats["direction_counts"]["LONG"] == 3
            assert stats["direction_counts"]["SHORT"] == 2

    def test_get_signal_stats_empty_db(self):
        """測試空資料庫的統計"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.db_path):
            signals_db.init_db()

            stats = signals_db.get_signal_stats(symbol="NONEXISTENT", days=30)
            assert stats["total_signals"] == 0


class TestDbConnection:
    """測試資料庫連線管理"""

    def test_connection_context_manager_commits(self):
        """測試 context manager 自動 commit"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.temp_db.name):
            signals_db.init_db()
            with signals_db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO signals (symbol, signal, direction, price, strategy, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    ("BTCUSDT", "BUY", "LONG", 50000, "Test", datetime.now().isoformat()),
                )

            # 重新連線驗證已寫入
            with signals_db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM signals")
                count = cursor.fetchone()[0]
                assert count == 1

    def test_connection_context_manager_rollback_on_error(self):
        """測試錯誤時 rollback"""
        import signals_db

        with patch("signals_db.get_db_path", return_value=self.temp_db.name):
            signals_db.init_db()

            try:
                with signals_db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO signals (symbol, signal, direction, price, strategy, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                        ("BTCUSDT", "BUY", "LONG", 50000, "Test", datetime.now().isoformat()),
                    )
                    raise ValueError("Simulated error")
            except ValueError:
                pass

            # 驗證資料已 rollback
            with signals_db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM signals")
                count = cursor.fetchone()[0]
                assert count == 0
