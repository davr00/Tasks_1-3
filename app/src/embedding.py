import uuid
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, set_key
from pathlib import Path
from typing import Dict, Optional, List

from app.config import settings
from app.logger import logger

ENV_PATH = Path(".env")

_current_model_id: Optional[uuid.UUID] = None
_current_model: Optional[SentenceTransformer] = None


def get_models() -> Dict[uuid.UUID, str]:
    load_dotenv(dotenv_path=ENV_PATH)

    env_models = {
        key.removeprefix("EMBEDDING_MODEL_"): value
        for key, value in os.environ.items()
        if key.startswith("EMBEDDING_MODEL_")
    }

    updated = False
    model_registry: Dict[uuid.UUID, str] = {}

    for model_name in settings.EMBEDDING_MODEL_NAMES:
        env_key = f"EMBEDDING_MODEL_{model_name}"

        if model_name not in env_models:
            new_uuid = str(uuid.uuid4())
            set_key(dotenv_path=ENV_PATH, key_to_set=env_key, value_to_set=new_uuid)
            logger.info(f"Новая модель добавлена в .env: {model_name} = {new_uuid}")
            model_uuid = new_uuid
            updated = True
        else:
            model_uuid = env_models[model_name]

        try:
            model_registry[uuid.UUID(model_uuid)] = model_name
        except ValueError:
            raise ValueError(f"Некорректный UUID в .env для модели '{model_name}': {model_uuid}")

    if updated:
        logger.info(".env файл обновлён")

    return model_registry


def get_embedding(model_id: uuid.UUID, texts: List[str]) -> List[List[float]]:
    global _current_model_id, _current_model

    model_registry = get_models()

    if model_id not in model_registry:
        raise ValueError(f"Модель с ID {model_id} не найдена в .env")

    model_name = model_registry[model_id]

    if _current_model_id != model_id:
        logger.info(f"Переключение на новую модель: '{model_name}' [{model_id}]")

        _current_model = None
        _current_model_id = None
        torch.cuda.empty_cache()

        _current_model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        _current_model_id = model_id

    logger.debug(f"Создаются эмбеддинги для {len(texts)} текстов с моделью '{model_name}'")
    vectors = _current_model.encode(texts, normalize_embeddings=True).tolist()
    torch.cuda.empty_cache()

    return vectors


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return float(dot_product / (norm1 * norm2))


def angular_similarity(vec1, vec2):
    cos_sim = cosine_similarity(vec1, vec2)
    return float(np.arccos(cos_sim) / np.pi)
