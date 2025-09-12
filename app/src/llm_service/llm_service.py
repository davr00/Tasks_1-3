import gc
import re
import time
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.config import settings
from app.logger import logger


MODEL_NAME = settings.LLM_MODEL_NAME
VLLM_URL = settings.VLLM_URL


class LLMService:
    def __init__(self, model_name=MODEL_NAME, vllm_url=VLLM_URL):
        self.vllm_url = vllm_url + "/v1/chat/completions"
        self.model = model_name
        self.temperature = 0.1
        self.hf_model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, cache_dir=settings.CACHE_DIR)

    def _contains_chinese(self, text: str) -> bool:
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(chinese_pattern.search(text))

    def _load_model(self):
        if self.hf_model is None:
             logger.debug(f"Загрузка модели {self.model}...")
             self.hf_model = AutoModelForCausalLM.from_pretrained(
                 self.model,
                 torch_dtype="auto",
                 device_map="cuda",
                 cache_dir=settings.CACHE_DIR,
                 low_cpu_mem_usage=True,
             )
             logger.debug("Модель загружена")

    def _unload_model(self):
        if self.hf_model is not None:
            logger.debug(f"Выгрузка модели {self.model} из VRAM...")
            del self.hf_model
            self.hf_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.debug("Модель выгружена из VRAM.")

    def generate_with_vllm(self, prompt: str):
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

        return None

    def generate_with_tf(self, prompt: str, text: str, max_new_tokens: int = 16384) -> str:
        try:
            self._load_model()

            if self.hf_model is None or self.tokenizer is None:
                 raise RuntimeError("Модель или токенизатор не загружены")

            messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.hf_model.device)

            with torch.no_grad():
                generated_ids = self.hf_model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens
                )

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            self._unload_model()

            return output_text

        except Exception as e:
            logger.error(f"Ошибка при генерации через HF: {e}")
            return "(Ошибка при генерации ответа.)"

    def generate(self, prompt: str, text: str) -> str:
        prompt_for_vllm = f"""{prompt}\nИсходный текст: {text}"""
        answer = self.generate_with_vllm(prompt_for_vllm)
        if answer is not None:
            return answer

        logger.info("Переход на HuggingFace, так как vLLM недоступен или вернул ошибку.")
        answer = self.generate_with_tf(prompt=prompt, text=text)
        if answer is not None:
            return answer

        return "(Не удалось получить корректный ответ ни от vLLM, ни от HF.)"
