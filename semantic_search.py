from typing import List, Dict, Any
from elasticsearch import AsyncElasticsearch
from colpali_processor import ColPaliProcessor
import numpy as np


class SemanticSearchService:
    def __init__(self, config):
        self.config = config
        self.es = AsyncElasticsearch([config.ELASTICSEARCH_URL])
        self.index_name = config.ELASTICSEARCH_INDEX
        self.colpali = ColPaliProcessor(config)

    async def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Ð¡ÐµÐ¼Ð°Ð½Ñ‚Ð¸Ñ‡ÐµÑÐºÐ¸Ð¹ Ð¿Ð¾Ð¸ÑÐº Ñ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·Ð¾Ð²Ð°Ð½Ð¸ÐµÐ¼ ColPali"""
        if not query:
            return []

        try:
            # ÐŸÑ€Ð¾Ð²ÐµÑ€ÐºÐ° Ð¿Ð¾Ð´ÐºÐ»ÑŽÑ‡ÐµÐ½Ð¸Ñ Ðº ElasticSearch
            if not await self.es.ping():
                raise ConnectionError("Cannot connect to ElasticSearch")

            # Ð£Ð»ÑƒÑ‡ÑˆÐµÐ½Ð¸Ðµ Ð·Ð°Ð¿Ñ€Ð¾ÑÐ° Ñ Ð¿Ð¾Ð¼Ð¾Ñ‰ÑŒÑŽ ColPali
            enhanced_query = await self.colpali.enhance_search_query(query)

            # Ð¤Ð¾Ñ€Ð¼Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ðµ Ð¿Ð¾Ð¸ÑÐºÐ¾Ð²Ð¾Ð³Ð¾ Ð·Ð°Ð¿Ñ€Ð¾ÑÐ° Ð´Ð»Ñ ElasticSearch
            search_body = {
                'size': limit * 2,  # ÐŸÐ¾Ð»ÑƒÑ‡Ð°ÐµÐ¼ Ð±Ð¾Ð»ÑŒÑˆÐµ Ñ€ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚Ð¾Ð² Ð´Ð»Ñ Ð¿Ð¾ÑÐ»ÐµÐ´ÑƒÑŽÑ‰ÐµÐ³Ð¾ Ð¿ÐµÑ€ÐµÑ€Ð°Ð½Ð¶Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ñ
                'query': {
                    'bool': {
                        'should': [
                            # Ð¢ÐµÐºÑÑ‚Ð¾Ð²Ñ‹Ð¹ Ð¿Ð¾Ð¸ÑÐº Ð¿Ð¾ ÑƒÐ»ÑƒÑ‡ÑˆÐµÐ½Ð½Ð¾Ð¼Ñƒ Ð·Ð°Ð¿Ñ€Ð¾ÑÑƒ
                            {
                                'multi_match': {
                                    'query': enhanced_query['enhanced_query'],
                                    'fields': ['content^2', 'metadata.*'],
                                    'type': 'best_fields',
                                    'boost': 0.3
                                }
                            },
                            # Ð’ÐµÐºÑ‚Ð¾Ñ€Ð½Ñ‹Ð¹ Ð¿Ð¾Ð¸ÑÐº Ð¿Ð¾ ÑÐ¼Ð±ÐµÐ´Ð´Ð¸Ð½Ð³Ð°Ð¼
                            {
                                'script_score': {
                                    'query': {'match_all': {}},
                                    'script': {
                                        'source': "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",
                                        'params': {
                                            'query_vector': enhanced_query['embeddings']
                                        }
                                    },
                                    'boost': 0.7
                                }
                            }
                        ]
                    }
                },
                'highlight': {
                    'fields': {
                        'content': {
                            'number_of_fragments': 3,
                            'fragment_size': 150
                        }
                    }
                }
            }

            # Ð’Ñ‹Ð¿Ð¾Ð»Ð½ÐµÐ½Ð¸Ðµ Ð¿Ð¾Ð¸ÑÐºÐ°
            response = await self.es.search(
                index=self.index_name,
                body=search_body
            )

            # ÐžÐ±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ° Ñ€ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚Ð¾Ð²
            results = self._process_search_results(response)

            # ÐŸÐµÑ€ÐµÑ€Ð°Ð½Ð¶Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ðµ Ñ Ð¿Ð¾Ð¼Ð¾Ñ‰ÑŒÑŽ ColPali
            reranked_results = await self.colpali.rerank_results(query, results)

            return reranked_results[:limit]

        except Exception as e:
            print(f"Semantic search error: {str(e)}")
            return []

    def _process_search_results(self, response: Dict) -> List[Dict[str, Any]]:
        """ÐžÐ±Ñ€Ð°Ð±Ð¾Ñ‚ÐºÐ° Ñ€ÐµÐ·ÑƒÐ»ÑŒÑ‚Ð°Ñ‚Ð¾Ð² Ð¿Ð¾Ð¸ÑÐºÐ°"""
        results = []

        for hit in response['hits']['hits']:
            result = {
                'id': hit['_id'],
                'content': hit['_source']['content'],
                'score': hit['_score'],
                'metadata': hit['_source'].get('metadata', {}),
                'highlights': hit.get('highlight', {}).get('content', []),
                'document_type': hit['_source'].get('document_type', 'unknown')
            }
            results.append(result)

        return results