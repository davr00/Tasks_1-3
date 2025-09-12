import asyncio
from typing import Optional, List
from uuid import UUID

import torch
from langdetect import detect, LangDetectException
from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse

from app.src.embedding import get_models, get_embedding
from app.src.llm_service.prompts import normalize_text
from app.logger import logger
from app.schema import EmbeddingResponse, ModelResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_QUEUE_SIZE = 3
queue = asyncio.Queue(MAX_QUEUE_SIZE)
MAX_TEXT_SIZE = 10 * 1024 * 1024




@app.get("/api/models", response_model=List[ModelResponse])
def list_models():
    try:
        logger.debug("Получение списка моделей")
        model_registry = get_models()
        logger.debug(f"Модели: {len(model_registry)}")
        torch.cuda.empty_cache()

        return [
            {"model_id": model_id, "model_name": model_name[0], "dimension": model_name[1]}
            for model_id, model_name in model_registry.items()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении списка моделей: {e}")


@app.post("/api/normalize", response_class=PlainTextResponse)
async def normalize(text: str = Body(..., media_type="text/plain")):
    logger.info(f"Запрос на нормализацию. Текущая очередь: {queue.qsize()}/{MAX_QUEUE_SIZE}")

    if queue.full():
        logger.exception("Очередь переполнена. Возвращаем 429")
        raise HTTPException(status_code=429, detail="Слишком много запросов. Попробуйте позже.")

    if len(text.encode("utf-8")) > MAX_TEXT_SIZE:
        logger.exception(f"Превышен лимит текста ({len(text.encode('utf-8'))} байт > {MAX_TEXT_SIZE})")
        raise HTTPException(
            status_code=413,
            detail="Payload Too Large: текст превышает допустимый размер 10MB"
        )

    fut = asyncio.get_event_loop().create_future()
    await queue.put(fut)

    try:
        text.encode("utf-8")
        normalized_text = await normalize_text(text)
        logger.info(f"Нормализация завершена. Очередь после выполнения: {queue.qsize()}/{MAX_QUEUE_SIZE}")
        torch.cuda.empty_cache()

        try:
            lang = detect(normalized_text)
        except LangDetectException:
            lang = "und"

        return PlainTextResponse(
            content=normalized_text,
            media_type="text/plain; charset=utf-8",
            headers={"language": lang}
        )

    except UnicodeEncodeError:
        logger.exception("Неподдерживаемая кодировка входного текста")
        raise HTTPException(
            status_code=400,
            detail="INVALID_ENCODING: неподдерживаемая кодировка. Ожидается UTF-8."
        )
    except HTTPException as e:
        raise e
    except Exception:
        logger.exception("Ошибка сервера при нормализации текста")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await queue.get()
        queue.task_done()


@app.post("/api/embedding", response_model=EmbeddingResponse)
async def embedding(
        text: str = Body(..., media_type="text/plain"),
        model_id: Optional[UUID] = Header(None, alias="x-model-id")
):
    logger.info(f"Запрос на эмбеддинг. Текущая очередь: {queue.qsize()}/{MAX_QUEUE_SIZE}")

    if queue.full():
        raise HTTPException(
            status_code=429,
            detail="Слишком много запросов. Попробуйте позже."
        )

    fut = asyncio.get_event_loop().create_future()
    await queue.put(fut)

    try:
        logger.info(f"Text: {text[:100]}...")
        logger.info(f"model_id: {model_id}")
        model_registry = get_models()

        if model_id is None:
            if model_registry:
                model_id, default_model_name = next(iter(model_registry.items()))
                logger.info(f"Выбрана модель {default_model_name}")
            else:
                raise HTTPException(status_code=400, detail="Missing 'model-id' header")

        if model_id not in model_registry:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        normalized_text = await normalize_text(text)
        logger.info(f"Нормализация завершена. Очередь после выполнения: {queue.qsize()}/{MAX_QUEUE_SIZE}")
        torch.cuda.empty_cache()
        emb, dim = get_embedding(model_id, normalized_text)
        torch.cuda.empty_cache()

        result = EmbeddingResponse(
            embeddings=emb,
            dimension=dim
        )

        return JSONResponse(
            content=result.model_dump(),
            media_type="application/json"
        )

    except HTTPException as e:
        raise e
    except Exception:
        logger.exception("Ошибка сервера при обработке запроса /embedding")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await queue.get()
        queue.task_done()
