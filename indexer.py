import faiss
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from PIL import Image


class DocumentIndexer:
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        self.init_indices()

    def init_indices(self):
        try:
            self.unified_index = faiss.IndexFlatL2(self.config['colpalii_embedding_dim'])
            self.load_indices()
            self.logger.info("Indices initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing indices: {str(e)}")
            raise

    def load_indices(self):
        unified_index_path = Path(self.config['index_dir']) / 'unified_index.faiss'
        if unified_index_path.exists():
            self.unified_index = faiss.read_index(str(unified_index_path))

    def save_indices(self):
        index_dir = Path(self.config['index_dir'])
        index_dir.mkdir(exist_ok=True)
        faiss.write_index(self.unified_index, str(index_dir / 'unified_index.faiss'))

    def index_document(self, document_path: Path, metadata: dict) -> bool:
        try:
            doc_type = document_path.suffix.lower()

            if doc_type in ['.jpg', '.jpeg', '.png', '.bmp']:
                image = Image.open(document_path)
                embedding = self.model_manager.process_with_colpalii(image, 'image')
            elif doc_type in ['.txt', '.doc', '.docx', '.pdf']:
                with open(document_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                embedding = self.model_manager.process_with_colpalii(text, 'text')

            self.unified_index.add(embedding)
            self.save_metadata(document_path, metadata, embedding)

            self.logger.info(f"Successfully indexed document: {document_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error indexing document {document_path}: {str(e)}")
            return False

    def save_metadata(self, document_path: Path, metadata: dict, embedding: np.ndarray):
        metadata_path = Path(self.config['metadata_dir']) / f"{document_path.stem}.json"
        metadata.update({
            'path': str(document_path),
            'embedding_id': self.unified_index.ntotal - 1,
            'indexed_at': datetime.now().isoformat()
        })
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)