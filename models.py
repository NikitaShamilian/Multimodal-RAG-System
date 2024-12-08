import torch
from transformers import AutoProcessor, AutoModel
from sentence_transformers import SentenceTransformer
import logging

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.init_models()

    def init_models(self):
        try:
            self.colpalii_processor = AutoProcessor.from_pretrained("microsoft/ColPalii-17B")
            self.colpalii_model = AutoModel.from_pretrained("microsoft/ColPalii-17B")
            self.text_model = SentenceTransformer(self.config['text_model'])
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.colpalii_model.to(self.device)
            self.logger.info("Models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise