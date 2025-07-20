import re
import time
import requests

from app.config import settings
from app.logger import logger


MODEL_NAME = settings.LLM_MODEL_NAME
VLLM_URL = settings.VLLM_URL


class LLMService:
    def __init__(self, model_name=MODEL_NAME, vllm_url=VLLM_URL):

        self.vllm_url = vllm_url + "/v1/chat/completions"
        self.model = model_name
        self.temperature = 0

    def _contains_chinese(self, text: str) -> bool:
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(chinese_pattern.search(text))

    def generate_with_llm(self, prompt: str):
        message = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": 32768,
            "chat_template_kwargs": {"enable_thinking": False}
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url=self.vllm_url, json=message)

                if response.status_code == 200:
                    result = response.json()
                    answer = result["choices"][0]["message"]["content"]
                    if not self._contains_chinese(answer):
                        return answer
                else:
                    logger.debug(f"Получен ответ с китайскими символами. Попытка {attempt + 1} из {max_retries}. Повторяем запрос...")
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Ошибка при получении ответа: {e}")

        return "(Не удалось получить корректный ответ. Возможно, модель возвращает нежелательные символы.)"
