"""系統設定載入"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# 載入 .env 檔案
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


class Settings(BaseSettings):
    """應用程式設定"""
    
    # Binance API
    binance_api_key: str = Field(default="", alias="BINANCE_API_KEY")
    binance_api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    
    # Discord
    discord_webhook_url: str = Field(default="", alias="DISCORD_WEBHOOK_URL")
    
    # PostgreSQL
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="crypto_platform", alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    
    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    
    # 系統
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    @property
    def postgres_dsn(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
