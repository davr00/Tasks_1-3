import gc
import random
import uuid
import os

import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, set_key
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from app.config import settings
from app.logger import logger

ENV_PATH = Path(".env")

_current_model_id: Optional[uuid.UUID] = None
_current_model: Optional[SentenceTransformer] = None


def get_model_dimension(model_name: str) -> int:
    model = SentenceTransformer(model_name, device="cpu", cache_folder=settings.CACHE_DIR)
    return model.get_sentence_embedding_dimension()


def parse_env_value(value: str) -> Tuple[str, int]:
    try:
        parts = dict(item.split(":") for item in value.split(";"))
        return parts["uuid"], int(parts["dim"])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Некорректная запись в .env: {value}") from e


def format_env_value(model_uuid: str, dim: int) -> str:
    return f"uuid:{model_uuid};dim:{dim}"


def get_models() -> Dict[uuid.UUID, Tuple[str, int]]:
    load_dotenv(dotenv_path=ENV_PATH)

    env_models = {
        key.removeprefix("EMBEDDING_MODEL_"): value
        for key, value in os.environ.items()
        if key.startswith("EMBEDDING_MODEL_")
    }

    updated = False
    model_registry: Dict[uuid.UUID, Tuple[str, int]] = {}

    for model_name in settings.EMBEDDING_MODEL_NAMES:
        env_key = f"EMBEDDING_MODEL_{model_name}"

        if model_name not in env_models:
            new_uuid = str(uuid.uuid4())
            dim = get_model_dimension(model_name)
            value = format_env_value(new_uuid, dim)

            set_key(dotenv_path=ENV_PATH, key_to_set=env_key, value_to_set=value)
            logger.info(f"Новая модель добавлена в .env: {model_name} = {value}")
            model_uuid = new_uuid
            updated = True
        else:
            model_uuid, dim = parse_env_value(env_models[model_name])

        try:
            model_registry[uuid.UUID(model_uuid)] = (model_name, dim)
        except ValueError:
            raise ValueError(f"Некорректный UUID в .env для модели '{model_name}': {model_uuid}")

    if updated:
        logger.info(".env файл обновлён")
    return model_registry

def _unload_current_model():
    global _current_model_id, _current_model
    if _current_model is not None:
        logger.debug(f"Выгрузка модели '{_current_model_id}' из памяти...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del _current_model
        _current_model = None
        _current_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Модель выгружена.")

def get_embedding(model_id: uuid.UUID, text: str) -> Tuple[List[float], int]:
    global _current_model_id, _current_model

    model_registry = get_models()

    if model_id not in model_registry:
        raise ValueError(f"Модель с ID {model_id} не найдена в .env")

    model_name, dim = model_registry[model_id]

    if _current_model_id != model_id:
        logger.info(f"Переключение на новую модель: '{model_name}' [{model_id}]")
        _unload_current_model()

        logger.debug(f"Загрузка модели '{model_name}'...")
        try:
            _current_model = SentenceTransformer(model_name, device="cuda:0", cache_folder=settings.CACHE_DIR)
            _current_model_id = model_id
            logger.debug(f"Модель '{model_name}' загружена на устройство '{_current_model.device}'.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели '{model_name}': {e}")
            _unload_current_model()
            raise

    if _current_model is None:
        raise RuntimeError("Не удалось загрузить модель для генерации эмбеддинга.")

    try:
        logger.debug(f"Создаются эмбеддинги для текста с моделью '{model_name}'...")
        vectors = _current_model.encode(text, show_progress_bar=False).tolist()
        logger.debug("Эмбеддинг успешно создан.")
        _unload_current_model()
        return vectors, dim
    except Exception as e:
        logger.error(f"Ошибка при генерации эмбеддинга моделью '{model_name}': {e}")
        raise
