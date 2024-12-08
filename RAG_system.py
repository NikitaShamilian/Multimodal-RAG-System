import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from models import ModelManager
from indexer import DocumentIndexer
from search import DocumentSearcher
from config import DEFAULT_CONFIG


class MultimodalRAGSystem:
    def __init__(self, config=None):
        self.config = config or {
            "index_dir": "./data/indices",
            "temp_dir": "./data/temp",
            "cache_dir": "./data/cache",
            "model_dir": "./data/models"
        }

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"rag_system_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        for dir_path in [self.config['temp_dir'],
                         self.config['index_dir'],
                         self.config['metadata_dir']]:
            os.makedirs(dir_path, exist_ok=True)

    def index_document(self, document_path: Path, metadata: Dict[str, Any]) -> bool:
        return self.indexer.index_document(document_path, metadata)

    def search(self, query, modality: str, k: int = 5):
        return self.searcher.search(query, modality, k)

    def save_indices(self):
        self.indexer.save_indices()