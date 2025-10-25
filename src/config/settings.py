from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AZURE_OPENAI_KEY: str
    AZURE_OPENAI_ENDPOINT: str 
    AZURE_OPENAI_DEPLOYMENT: str
    AZURE_OPENAI_API_VERSION: str
    CHROMA_DB_PATH: str = "./data/chroma_db"
    AZURE_OPENAI_KEY_EMBEDDING_MODEL: str
    AZURE_OPENAI_KEY_EMBEDDING: str

    class Config:
        env_file = ".env"

settings = Settings()