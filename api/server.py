"""API 서버 실행 진입점."""

import uvicorn

from api import create_app
from config import load_config


def main() -> None:
    settings = load_config()
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
