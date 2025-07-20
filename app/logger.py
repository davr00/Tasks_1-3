import logging
import sys
import os
from app.config import settings


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[41m",
        "RESET": "\033[0m"
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{self.COLORS['RESET']}"


def setup_logger(name="rag_system", level=logging.DEBUG):
    logger = logging.getLogger(name)

    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = "%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] %(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(os.path.join(settings.LOG_DIR, "text_normalize.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(log_format))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Логгер успешно настроен")
    return logger

logger = setup_logger()
