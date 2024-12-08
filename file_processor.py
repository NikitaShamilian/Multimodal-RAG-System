from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import numpy as np
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все источники (в продакшене лучше указать конкретные домены)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],  # Разрешить все заголовки
)
#app = FastAPI()


class FileProcessor:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.supported_formats = {
            'text': ['.txt', '.doc', '.docx', '.pdf'],
            'image': ['.jpg', '.jpeg', '.png'],
            'video': ['.mp4', '.avi', '.mov'],
            'audio': ['.mp3', '.wav']
        }

    async def process_file(self, file: UploadFile) -> dict:
        # Получаем расширение файла
        ext = os.path.splitext(file.filename)[1].lower()

        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            if ext in self.supported_formats['text']:
                return await self.process_text(temp_path)
            elif ext in self.supported_formats['image']:
                return await self.process_image(temp_path)
            elif ext in self.supported_formats['video']:
                return await self.process_video(temp_path)
            elif ext in self.supported_formats['audio']:
                return await self.process_audio(temp_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        finally:
            # Удаляем временный файл
            os.unlink(temp_path)

    async def process_text(self, file_path: str) -> dict:
        # Обработка текстовых файлов
        pass

    async def process_image(self, file_path: str) -> dict:
        # Обработка изображений
        pass

    async def process_video(self, file_path: str) -> dict:
        # Обработка видео
        pass

    async def process_audio(self, file_path: str) -> dict:
        # Обработка аудио
        pass


# Инициализация процессора файлов
file_processor = FileProcessor("bert-base-uncased")


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        result = await file_processor.process_file(file)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )


@app.post("/batch-analyze")
async def analyze_multiple_files(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            result = await file_processor.process_file(file)
            results.append({
                "filename": file.filename,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6065)