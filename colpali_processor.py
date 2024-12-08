from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import logging
from torch.cuda.amp import autocast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColPaliProcessor:
    def __init__(self, config):
        self.config = config
        self.model_name = config.COLPALI_MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def enhance_search_query(self, query: str) -> Dict[str, Any]:
        """Ð£Ð»ÑƒÑ‡ÑˆÐµÐ½Ð¸Ðµ Ð¿Ð¾Ð¸ÑÐºÐ¾Ð²Ð¾Ð³Ð¾ Ð·Ð°Ð¿Ñ€Ð¾ÑÐ°"""
        try:
            prompt = f"Enhance and expand the following search query: {query}\nEnhanced query:"
            inputs = self._tokenize(prompt)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )

            enhanced_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            enhanced_query = enhanced_query.replace(prompt, "").strip()

            embeddings = await self._create_embeddings(enhanced_query)

            return {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "embeddings": embeddings
            }
        except Exception as e:
            logger.error(f"Query enhancement error: {str(e)}")
            return {
                "original_query": query,
                "enhanced_query": query,
                "embeddings": await self._create_embeddings(query)
            }

    async def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ÐŸÐµÑ€ÐµÑ€Ð°Ð½Ð¶Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ðµ Ñ€ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚Ð¾Ð²"""
        try:
            query_data = await self.enhance_search_query(query)
            query_embedding = np.array(query_data["embeddings"])

            for result in results:
                result_embedding = await self._create_embeddings(result["content"])
                similarity = self._compute_similarity(query_embedding, np.array(result_embedding))

                result["colpali_score"] = float(similarity)
                result["final_score"] = (
                        result.get("score", 0) * self.config.ORIGINAL_SCORE_WEIGHT +
                        similarity * self.config.SIMILARITY_SCORE_WEIGHT
                )

            results.sort(key=lambda x: x["final_score"], reverse=True)
            return results

        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            return results

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Ð¢Ð¾ÐºÐµÐ½Ð¸Ð·Ð°Ñ†Ð¸Ñ Ñ‚ÐµÐºÑÑ‚Ð°"""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.MAX_LENGTH
        ).to(self.device)

    @autocast()
    async def _create_embeddings(self, text: str) -> List[float]:
        """Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ ÑÐ¼Ð±ÐµÐ´Ð´Ð¸Ð½Ð³Ð¾Ð²"""
        try:
            inputs = self._tokenize(text)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1].mean(dim=1)
            return embeddings.cpu().numpy().tolist()[0]
        except Exception as e:
            logger.error(f"Embedding creation error: {str(e)}")
            return []

    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Ð’Ñ‹Ñ‡Ð¸ÑÐ»ÐµÐ½Ð¸Ðµ ÐºÐ¾ÑÐ¸Ð½ÑƒÑÐ½Ð¾Ð³Ð¾ ÑÑ…Ð¾Ð´ÑÑ‚Ð²Ð°"""
        try:
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception:
            return 0.0