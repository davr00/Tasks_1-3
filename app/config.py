from typing import List

from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    LOG_DIR: str = "log"

    EMBEDDING_MODEL_NAMES: List[str] = [
        "ai-forever/FRIDA",
        "google/embeddinggemma-300m",
        "Qwen/Qwen3-Embedding-0.6B"
    ]

    LLM_MODEL_NAME: str = "Qwen/Qwen3-8B-AWQ"
    # LLM_MODEL_NAME: str = "Qwen/Qwen3-4B-Instruct-2507"

    VLLM_URL: str = "http://localhost:8080"

    CACHE_DIR: str = "models"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

os.makedirs(settings.LOG_DIR, exist_ok=True)
