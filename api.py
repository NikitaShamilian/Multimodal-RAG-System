from pathlib import Path
import json
from datetime import datetime
from typing import List, Optional
import asyncio
import aiofiles
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
import uvicorn
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все origins
    allow_credentials=False,  # Должно быть False при allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Константы для загрузки файлов
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_CONCURRENT_UPLOADS = 5
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.doc', '.docx', '.json', '.csv'}


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    filter_criteria: Optional[dict] = None


class DocumentIndexer:
    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)
        self.documents = {}
        self.embeddings = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    async def index_document(self, file_path, metadata):
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # Выполняем тяжелые вычисления в отдельном потоке
            embedding = await asyncio.to_thread(self.model.encode, content)

            doc_id = str(len(self.documents))
            self.documents[doc_id] = {
                'content': content,
                'metadata': metadata,
                'embedding': embedding
            }
            return True, doc_id
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return False, None


class MultimodalRAGSystem:
    def __init__(self, config, save_dir: str = "data"):
        self.config = config
        self.indexer = DocumentIndexer(config['index_dir'])
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data = {
            "version": "1.0",
            "last_updated": None,
            "documents": [],
            "embeddings": []
        }

    async def index_document(self, file_path, metadata):
        try:
            success, doc_id = await self.indexer.index_document(file_path, metadata)
            if success:
                self.data["documents"].append({
                    "id": doc_id,
                    "metadata": metadata,
                    "path": str(file_path)
                })
            return success, doc_id
        except Exception as e:
            logger.error(f"Error in index_document: {str(e)}")
            return False, None

    def search(self, query, k=5, filter_criteria=None):
        try:
            if not self.indexer.documents:
                return []

            query_embedding = self.indexer.model.encode(query)

            similarities = []
            for doc_id, doc in self.indexer.documents.items():
                # Применяем фильтры, если они есть
                if filter_criteria:
                    if not all(doc['metadata'].get(key) == value
                               for key, value in filter_criteria.items()):
                        continue

                similarity = np.dot(query_embedding, doc['embedding']) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc['embedding'])
                )
                similarities.append((doc_id, float(similarity)))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k = similarities[:k]

            results = []
            for doc_id, score in top_k:
                doc = self.indexer.documents[doc_id]
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': score
                })

            return results
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    async def save_indices(self):
        try:
            self.data["last_updated"] = datetime.now().isoformat()
            save_path = self.save_dir / f"rag_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            async with aiofiles.open(save_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.data, indent=2, ensure_ascii=False))

            return {
                "saved_documents": len(self.data["documents"]),
                "saved_embeddings": len(self.data["embeddings"]),
                "save_path": str(save_path),
                "version": self.data["version"],
                "timestamp": self.data["last_updated"]
            }
        except Exception as e:
            raise Exception(f"Error saving indices: {str(e)}")


def is_valid_file(filename: str, file_size: int) -> tuple[bool, str]:
    """Проверка валидности файла"""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type: {extension}"

    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024)}MB"

    return True, ""


base_dir = Path("./data")
config = {
    "index_dir": str(base_dir / "indices"),
    "temp_dir": str(base_dir / "temp"),
    "cache_dir": str(base_dir / "cache"),
    "model_dir": str(base_dir / "models"),
}

for dir_path in config.values():
    Path(dir_path).mkdir(parents=True, exist_ok=True)

app = FastAPI()
rag_system = MultimodalRAGSystem(config)


async def process_file(file: UploadFile, temp_dir: Path) -> dict:
    """Обработка отдельного файла"""
    temp_path = None
    try:
        # Создание временного пути для файла
        temp_path = temp_dir / f"{datetime.now().timestamp()}_{file.filename}"

        # Получение размера файла
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        # Проверка файла
        is_valid, error_message = is_valid_file(file.filename, file_size)
        if not is_valid:
            raise ValueError(error_message)

        # Сохранение файла
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        # Подготовка метаданных
        metadata_dict = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": file_size,
            "upload_time": datetime.now().isoformat(),
            "file_extension": Path(file.filename).suffix.lower()
        }

        # Индексация документа
        success, doc_id = await rag_system.index_document(temp_path, metadata_dict)

        if success:
            return {
                "status": "success",
                "message": f"File {file.filename} successfully uploaded and indexed",
                "metadata": metadata_dict,
                "doc_id": doc_id
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to index file {file.filename}")

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing file {file.filename}",
            "error": str(e)
        }
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


@app.post("/index/documents")
async def index_multiple_documents(files: List[UploadFile] = File(...)):
    """Эндпоинт для загрузки множества файлов"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    temp_dir = Path(config['temp_dir'])
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Создание семафора для ограничения параллельных загрузок
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

    async def process_with_semaphore(file):
        async with semaphore:
            return await process_file(file, temp_dir)

    # Создание задач для каждого файла
    tasks = [asyncio.create_task(process_with_semaphore(file)) for file in files]

    # Ожидание завершения всех задач
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Подготовка ответа
    response = {
        "total_files": len(files),
        "successful": sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success"),
        "failed": sum(1 for r in results if isinstance(r, dict) and r.get("status") == "error"),
        "results": results
    }

    return response


@app.post("/index/document")
async def index_single_document(file: UploadFile = File(...)):
    """Эндпоинт для загрузки одного файла"""
    temp_dir = Path(config['temp_dir'])
    temp_dir.mkdir(parents=True, exist_ok=True)

    result = await process_file(file, temp_dir)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return result


@app.post("/search")
async def search(request: SearchRequest):
    try:
        results = rag_system.search(
            request.query,
            k=request.k,
            filter_criteria=request.filter_criteria
        )
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save")
async def save_indices(background_tasks: BackgroundTasks):
    try:
        result = await rag_system.save_indices()
        return {
            "status": "success",
            "message": "Indices successfully saved",
            "details": result
        }
    except Exception as e:
        logger.error(f"Save error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6065)