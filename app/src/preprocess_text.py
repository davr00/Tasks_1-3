import base64
from fastapi import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_413_REQUEST_ENTITY_TOO_LARGE


MAX_TEXT_SIZE = 10 * 1024 * 1024


def decode_base64_text(b64_text: str) -> str:
    try:
        decoded_bytes = base64.b64decode(b64_text)
    except Exception:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="INVALID_ENCODING: Unable to decode base64."
        )

    if len(decoded_bytes) > MAX_TEXT_SIZE:
        raise HTTPException(
            status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="PAYLOAD_TOO_LARGE: Input text exceeds 10MB."
        )

    try:
        return decoded_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="INVALID_ENCODING: Expected UTF-8 text."
        )


def encode_base64_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")
