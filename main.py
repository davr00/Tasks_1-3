import time
import torch
from typing import Optional, List
from uuid import UUID

from langdetect import detect
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse

from app.src.embedding import get_models, get_embedding, cosine_similarity, angular_similarity
from app.src.llm_service.prompts import normalize_text
from app.logger import logger
from app.src.preprocess_text import decode_base64_text, encode_base64_text
from app.schema import NormalizeRequest, NormalizeResponse, EmbeddingRequest, EmbeddingResponse, ModelResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/models", response_model=List[ModelResponse])
def list_models():
    try:
        model_registry = get_models()
        return [
            {"model_id": model_id, "model_name": model_name}
            for model_id, model_name in model_registry.items()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении списка моделей: {e}")


@app.post("/api/normalize", response_model=NormalizeResponse)
async def normalize(request: NormalizeRequest):
    try:
        original_text = decode_base64_text(request.text)
        normalized_text = normalize_text(original_text)

        try:
            lang = detect(normalized_text)
        except Exception:
            lang = "und"

        result = NormalizeResponse(
            normalized_text=encode_base64_text(normalized_text)
        )

        return PlainTextResponse(
            content=result.model_dump_json(),
            media_type="text/plain; charset=utf-8",
            headers={"language": lang}
        )

    except HTTPException as e:
        raise e
    except Exception:
        logger.exception("Ошибка сервера при нормализации текста")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/embedding", response_model=EmbeddingResponse)
def embedding(request: EmbeddingRequest, model_id: Optional[UUID] = Header(None, alias="model-id")):
    try:
        model_registry = get_models()

        if model_id is None:
            if model_registry:
                model_id, default_model_name = next(iter(model_registry.items()))
                logger.info(f"Выбрана модель {default_model_name}")
            else:
                raise HTTPException(status_code=400, detail="Missing 'model-id' header")

        if model_id not in model_registry:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

        b64_text1 = request.text1
        b64_text2 = request.text2

        original_text1 = decode_base64_text(b64_text1)
        normalized_text1 = normalize_text(original_text1)

        original_text2 = decode_base64_text(b64_text2)
        normalized_text2 = normalize_text(original_text2)

        texts = [normalized_text1, normalized_text2]

        vectors = get_embedding(model_id, texts)

        emb1 = vectors[0]
        emb2 = vectors[1]

        cos_sim = cosine_similarity(emb1, emb2)
        ang_sim = angular_similarity(emb1, emb2)

        result = EmbeddingResponse(
            embedding1=emb1,
            embedding2=emb2,
            cosine_similarity=cos_sim,
            angular_similarity_radians=ang_sim,
            developer_value=None
        )

        torch.cuda.empty_cache()

        return JSONResponse(
            content=result.model_dump(),
            media_type="application/json"
        )

    except HTTPException as e:
        raise e
    except Exception:
        logger.exception("Ошибка сервера при обработке запроса /embedding")
        raise HTTPException(status_code=500, detail="Internal server error")
