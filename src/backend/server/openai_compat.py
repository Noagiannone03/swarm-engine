from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse


def openai_error_payload(
    message: str,
    *,
    err_type: str = "server_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code or err_type,
        }
    }


def openai_error_response(
    message: str,
    *,
    status_code: int,
    err_type: str = "server_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> JSONResponse:
    return JSONResponse(
        content=openai_error_payload(
            message,
            err_type=err_type,
            param=param,
            code=code,
        ),
        status_code=status_code,
    )
