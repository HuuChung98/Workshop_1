import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://aiportalapi.stu-platform.live/jpe")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "GPT-4o-mini")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    CHROMA_DB_PATH: str = "./data/chroma_db"
    AZURE_OPENAI_KEY_EMBEDDING_MODEL: str = os.getenv("AZURE_OPENAI_KEY_EMBEDDING_MODEL", "text-embedding-3-small")
    AZURE_OPENAI_KEY_EMBEDDING: str = os.getenv("AZURE_OPENAI_KEY_EMBEDDING")

    class Config:
        env_file = ".env"

settings = Settings()