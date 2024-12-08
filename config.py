from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Конфигурационные настройки приложения
    """
    # Название модели для ColPali
    COLPALI_MODEL_NAME: str = "gpt2"  # или другая подходящая модель

    # Параметры модели
    MAX_LENGTH: int = 512
    NUM_RETURN_SEQUENCES: int = 1
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9

    # Веса для ранжирования
    ORIGINAL_SCORE_WEIGHT: float = 0.3
    SIMILARITY_SCORE_WEIGHT: float = 0.7

    # Настройки устройства
    DEVICE: Optional[str] = None  # Автоматический выбор между cuda и cpu

    # Параметры логирования
    LOG_LEVEL: str = "INFO"

    # Модель для обработки изображений
    IMAGE_MODEL: str = "openai/clip-vit-base-patch32"

    # Пути для временных файлов
    TEMP_DIR: str = "temp"

    # Настройки Tesseract
    TESSERACT_CMD: Optional[str] = None  # Путь к исполняемому файлу tesseract

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
DEFAULT_CONFIG = {
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'index_path': 'data/search_index.pkl',
    'temp_dir': 'temp',
    'max_length': 512,
    'device': 'cpu',  # или 'cuda' если есть GPU
    'batch_size': 32
}
config = {
    'text_model': 'sentence-transformers/all-MiniLM-L6-v2'
}