from pydantic_settings import BaseSettings, SettingsConfigDict
#from pydantic import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    GENERATION_BACKEND: str
    VISION_BACKEND: str
    EMBEDDING_BACKEND : str 

    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None
    GROQ_API_KEY : str = None
    
    AZURE_OPENAI_API_KEY : str = None
    AZURE_OPENAI_ENDPOINT : str = None
    API_VERSION : str = None 

    GENERATION_MODEL_ID: str = None
    VISION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID :str = None

    INPUT_DAFAULT_MAX_CHARACTERS: int = None
    GENERATION_DAFAULT_MAX_TOKENS: int = None
    GENERATION_DAFAULT_TEMPERATURE: float = None
    
    VECTOR_DB_BACKEND : str
    VECTOR_DB_PATH : str
    DATASET :str
    DATABASE_SQL:str
    COLLECTION_NAME :str
    CLASSIFICATION_BACKEND :str
    CLASSIFICATION_MODEL_ID :str
    SQL_BACKEND :str
    SQL_MODEL_ID :str

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
