from typing import List, Dict, Union
from PIL import Image
from pathlib import Path
import json
import logging


class DocumentSearcher:
    def __init__(self, config, model_manager, indexer):
        self.config = config
        self.model_manager = model_manager
        self.indexer = indexer
        self.logger = logging.getLogger(__name__)

    def search(self, query: Union[str, Image.Image], modality: str, k: int = 5) -> List[Dict]:
        try:
            query_embedding = self.model_manager.process_with_colpalii(query, modality)
            D, I = self.indexer.unified_index.search(query_embedding, k)

            results = []
            for idx in I[0]:
                result = self.get_document_by_id(idx)
                if result:
                    results.append(result)
            return results
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []

    def get_document_by_id(self, idx: int) -> Dict:
        metadata_dir = Path(self.config['metadata_dir'])
        for metadata_file in metadata_dir.glob('*.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if metadata['embedding_id'] == idx:
                    return metadata
        return {}