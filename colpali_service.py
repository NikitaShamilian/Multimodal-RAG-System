from typing import List, Dict, Any, TypeVar, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from config import Settings
import logging
from torch.cuda.amp import autocast

# ÐÐ°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ° Ð»Ð¾Ð³Ð³ÐµÑ€Ð°
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Dict[str, Any])


class ColPaliException(Exception):
    """ÐŸÐ¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ñ‚ÐµÐ»ÑŒÑÐºÐ¾Ðµ Ð¸ÑÐºÐ»ÑŽÑ‡ÐµÐ½Ð¸Ðµ Ð´Ð»Ñ Ð¾ÑˆÐ¸Ð±Ð¾Ðº ColPali"""
    pass


class ColPaliService:
    def __init__(self, config: Settings):
        """
        Ð˜Ð½Ð¸Ñ†Ð¸Ð°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ñ ÑÐµÑ€Ð²Ð¸ÑÐ° ColPali

        Args:
            config: ÐžÐ±ÑŠÐµÐºÑ‚ ÐºÐ¾Ð½Ñ„Ð¸Ð³ÑƒÑ€Ð°Ñ†Ð¸Ð¸ Ñ Ð½Ð°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ°Ð¼Ð¸ Ð¼Ð¾Ð´ÐµÐ»Ð¸

        Raises:
            ValueError: Ð•ÑÐ»Ð¸ Ð½Ðµ Ð½Ð°ÑÑ‚Ñ€Ð¾ÐµÐ½Ð¾ Ð¸Ð¼Ñ Ð¼Ð¾Ð´ÐµÐ»Ð¸
            RuntimeError: Ð•ÑÐ»Ð¸ Ð½Ðµ ÑƒÐ´Ð°Ð»Ð¾ÑÑŒ Ð¸Ð½Ð¸Ñ†Ð¸Ð°Ð»Ð¸Ð·Ð¸Ñ€Ð¾Ð²Ð°Ñ‚ÑŒ Ð¼Ð¾Ð´ÐµÐ»ÑŒ
        """
        if not hasattr(config, 'COLPALI_MODEL_NAME') or not config.COLPALI_MODEL_NAME:
            raise ValueError("COLPALI_MODEL_NAME not configured")

        self.config = config
        try:
            self.model_name = config.COLPALI_MODEL_NAME
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # ÐÐ°ÑÑ‚Ñ€Ð¾Ð¹ÐºÐ° Ñ‚Ð¾ÐºÐµÐ½Ð¸Ð·Ð°Ñ‚Ð¾Ñ€Ð°
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            logger.info(f"ColPali initialized on device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ColPali model: {str(e)}")

    def _truncate_and_tokenize(self, text: str, max_length: int = 512) -> Optional[torch.Tensor]:
        """
        Ð¢Ð¾ÐºÐµÐ½Ð¸Ð·Ð°Ñ†Ð¸Ñ Ñ‚ÐµÐºÑÑ‚Ð° Ñ Ð¾Ð±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ¾Ð¹ Ð´Ð»Ð¸Ð½Ð½Ñ‹Ñ… Ð¿Ð¾ÑÐ»ÐµÐ´Ð¾Ð²Ð°Ñ‚ÐµÐ»ÑŒÐ½Ð¾ÑÑ‚ÐµÐ¹

        Args:
            text: Ð’Ñ…Ð¾Ð´Ð½Ð¾Ð¹ Ñ‚ÐµÐºÑÑ‚
            max_length: ÐœÐ°ÐºÑÐ¸Ð¼Ð°Ð»ÑŒÐ½Ð°Ñ Ð´Ð»Ð¸Ð½Ð° Ð¿Ð¾ÑÐ»ÐµÐ´Ð¾Ð²Ð°Ñ‚ÐµÐ»ÑŒÐ½Ð¾ÑÑ‚Ð¸ Ñ‚Ð¾ÐºÐµÐ½Ð¾Ð²

        Returns:
            Ð¢ÐµÐ½Ð·Ð¾Ñ€ Ñ‚Ð¾ÐºÐµÐ½Ð¾Ð² Ð¸Ð»Ð¸ None Ð² ÑÐ»ÑƒÑ‡Ð°Ðµ Ð¾ÑˆÐ¸Ð±ÐºÐ¸
        """
        try:
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            return None

    async def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Ð£Ð»ÑƒÑ‡ÑˆÐµÐ½Ð¸Ðµ Ð¿Ð¾Ð¸ÑÐºÐ¾Ð²Ð¾Ð³Ð¾ Ð·Ð°Ð¿Ñ€Ð¾ÑÐ°

        Args:
            query: Ð˜ÑÑ…Ð¾Ð´Ð½Ñ‹Ð¹ Ð¿Ð¾Ð¸ÑÐºÐ¾Ð²Ñ‹Ð¹ Ð·Ð°Ð¿Ñ€Ð¾Ñ

        Returns:
            Ð¡Ð»Ð¾Ð²Ð°Ñ€ÑŒ Ñ ÑƒÐ»ÑƒÑ‡ÑˆÐµÐ½Ð½Ñ‹Ð¼ Ð·Ð°Ð¿Ñ€Ð¾ÑÐ¾Ð¼ Ð¸ ÑÐ¼Ð±ÐµÐ´Ð´Ð¸Ð½Ð³Ð°Ð¼Ð¸

        Raises:
            ColPaliException: ÐŸÑ€Ð¸ Ð¾ÑˆÐ¸Ð±ÐºÐµ Ð¾Ð±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ¸ Ð·Ð°Ð¿Ñ€Ð¾ÑÐ°
        """
        if not query or not isinstance(query, str):
            raise ColPaliException("Invalid query format")

        try:
            prompt = f"Enhance and expand the following search query: {query}\nEnhanced query:"
            inputs = self._truncate_and_tokenize(prompt)

            if inputs is None:
                raise ColPaliException("Failed to tokenize input")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            enhanced_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            enhanced_query = enhanced_query.replace(prompt, "").strip()

            query_embeddings = await self.create_embeddings(enhanced_query)

            return {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "embeddings": query_embeddings
            }

        except Exception as e:
            logger.error(f"Query enhancement error: {str(e)}")
            return {
                "original_query": query,
                "enhanced_query": query,
                "embeddings": await self.create_embeddings(query)
            }

    @autocast()
    async def create_embeddings(self, text: str) -> List[float]:
        """
        Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ ÑÐ¼Ð±ÐµÐ´Ð´Ð¸Ð½Ð³Ð¾Ð² Ð´Ð»Ñ Ñ‚ÐµÐºÑÑ‚Ð°

        Args:
            text: Ð’Ñ…Ð¾Ð´Ð½Ð¾Ð¹ Ñ‚ÐµÐºÑÑ‚

        Returns:
            Ð¡Ð¿Ð¸ÑÐ¾Ðº ÑÐ¼Ð±ÐµÐ´Ð´Ð¸Ð½Ð³Ð¾Ð²
        """
        try:
            inputs = self._truncate_and_tokenize(text)
            if inputs is None:
                return []

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1].mean(dim=1)

            result = embeddings.cpu().numpy().tolist()[0]
            torch.cuda.empty_cache()  # ÐžÑ‡Ð¸ÑÑ‚ÐºÐ° CUDA ÐºÑÑˆÐ°
            return result

        except Exception as e:
            logger.error(f"Embedding creation error: {str(e)}")
            return []

    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Ð’Ñ‹Ñ‡Ð¸ÑÐ»ÐµÐ½Ð¸Ðµ ÐºÐ¾ÑÐ¸Ð½ÑƒÑÐ½Ð¾Ð³Ð¾ ÑÑ…Ð¾Ð´ÑÑ‚Ð²Ð° Ð¼ÐµÐ¶Ð´Ñƒ Ð²ÐµÐºÑ‚Ð¾Ñ€Ð°Ð¼Ð¸

        Args:
            vec1: ÐŸÐµÑ€Ð²Ñ‹Ð¹ Ð²ÐµÐºÑ‚Ð¾Ñ€
            vec2: Ð’Ñ‚Ð¾Ñ€Ð¾Ð¹ Ð²ÐµÐºÑ‚Ð¾Ñ€

        Returns:
            Ð—Ð½Ð°Ñ‡ÐµÐ½Ð¸Ðµ ÐºÐ¾ÑÐ¸Ð½ÑƒÑÐ½Ð¾Ð³Ð¾ ÑÑ…Ð¾Ð´ÑÑ‚Ð²Ð°
        """
        try:
            if vec1.size == 0 or vec2.size == 0:
                return 0.0

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(vec1, vec2) / (norm1 * norm2))

        except Exception as e:
            logger.error(f"Similarity computation error: {str(e)}")
            return 0.0

    async def rerank_results(self, query: str, results: List[T], top_k: int = 10) -> List[T]:
        """
        ÐŸÐµÑ€ÐµÑ€Ð°Ð½Ð¶Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ðµ Ñ€ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚Ð¾Ð² Ð¿Ð¾Ð¸ÑÐºÐ°

        Args:
            query: ÐŸÐ¾Ð¸ÑÐºÐ¾Ð²Ñ‹Ð¹ Ð·Ð°Ð¿Ñ€Ð¾Ñ
            results: Ð¡Ð¿Ð¸ÑÐ¾Ðº Ñ€ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚Ð¾Ð²
            top_k: ÐšÐ¾Ð»Ð¸Ñ‡ÐµÑÑ‚Ð²Ð¾ Ð²Ð¾Ð·Ð²Ñ€Ð°Ñ‰Ð°ÐµÐ¼Ñ‹Ñ… Ñ€ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚Ð¾Ð²

        Returns:
            ÐžÑ‚ÑÐ¾Ñ€Ñ‚Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð½Ñ‹Ð¹ ÑÐ¿Ð¸ÑÐ¾Ðº Ñ€ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚Ð¾Ð²
        """
        try:
            query_data = await self.enhance_query(query)
            query_embedding = np.array(query_data["embeddings"])

            for result in results:
                result_embedding = await self.create_embeddings(result["content"])
                similarity = self._compute_similarity(query_embedding, np.array(result_embedding))

                result["colpali_score"] = float(similarity)
                result["final_score"] = (
                        result.get("score", 0) * 0.3 +
                        similarity * 0.7
                )

            results.sort(key=lambda x: x["final_score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Reranking error: {str(e)}")
            return results[:top_k]

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð¿Ñ€Ð¾Ð¼Ð¿Ñ‚Ð° Ð´Ð»Ñ Ð³ÐµÐ½ÐµÑ€Ð°Ñ†Ð¸Ð¸ Ð¾Ñ‚Ð²ÐµÑ‚Ð°

        Args:
            query: Ð’Ð¾Ð¿Ñ€Ð¾Ñ
            context: ÐšÐ¾Ð½Ñ‚ÐµÐºÑÑ‚

        Returns:
            ÐŸÐ¾Ð´Ð³Ð¾Ñ‚Ð¾Ð²Ð»ÐµÐ½Ð½Ñ‹Ð¹ Ð¿Ñ€Ð¾Ð¼Ð¿Ñ‚
        """
        return (
            f"Context: {context}\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def _clean_generated_text(self, text: str, prompt: str) -> str:
        """
        ÐžÑ‡Ð¸ÑÑ‚ÐºÐ° ÑÐ³ÐµÐ½ÐµÑ€Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð½Ð¾Ð³Ð¾ Ñ‚ÐµÐºÑÑ‚Ð°

        Args:
            text: Ð¡Ð³ÐµÐ½ÐµÑ€Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð½Ñ‹Ð¹ Ñ‚ÐµÐºÑÑ‚
            prompt: Ð˜ÑÑ…Ð¾Ð´Ð½Ñ‹Ð¹ Ð¿Ñ€Ð¾Ð¼Ð¿Ñ‚

        Returns:
            ÐžÑ‡Ð¸Ñ‰ÐµÐ½Ð½Ñ‹Ð¹ Ñ‚ÐµÐºÑÑ‚
        """
        return text.replace(prompt, "").strip()

    async def generate_answer(self, query: str, context: str) -> str:
        """
        Ð“ÐµÐ½ÐµÑ€Ð°Ñ†Ð¸Ñ Ð¾Ñ‚Ð²ÐµÑ‚Ð° Ð½Ð° Ð¾ÑÐ½Ð¾Ð²Ðµ ÐºÐ¾Ð½Ñ‚ÐµÐºÑÑ‚Ð°

        Args:
            query: Ð’Ð¾Ð¿Ñ€Ð¾Ñ
            context: ÐšÐ¾Ð½Ñ‚ÐµÐºÑÑ‚

        Returns:
            Ð¡Ð³ÐµÐ½ÐµÑ€Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð½Ñ‹Ð¹ Ð¾Ñ‚Ð²ÐµÑ‚
        """
        if not query or not context:
            return "Invalid input parameters"

        try:
            prompt = self._create_prompt(query, context)
            inputs = self._truncate_and_tokenize(prompt)

            if inputs is None:
                return "Failed to process input"

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_generated_text(answer, prompt)

        except Exception as e:
            logger.error(f"Answer generation error: {str(e)}")
            return "Sorry, I couldn't generate an answer."