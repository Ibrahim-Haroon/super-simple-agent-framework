from os import getenv as env, path
from dotenv import load_dotenv
from src.util.singleton import singleton


@singleton
class Env:
    def __init__(self):
        load_dotenv(
            dotenv_path=path.join(path.dirname(path.realpath(__file__)),
                                  "../..", ".env")
        )
        self.__env: dict[str, str] = {}

    def __getitem__(self, key: str) -> str:
        if key not in self.__env:
            self.__env[key] = env(key)
        return self.__env[key]
