from ergodic.assets.asset_client import AssetClient
from ergodic.collections.collections_client import CollectionsClient
from ergodic.auth import get_health_check, login_to_api
import logging
import os

from dotenv import load_dotenv


load_dotenv()


class ErgodicClient:
    def __init__(self, api_url: str, username: str, password: str):
        self.token = self.login(api_url, username, password)
        self.api_url = api_url
        self.username = username
        # self.token = self.login(api_url, username, password)

        self.assets = AssetClient(api_url, self.token)
        self.collections = CollectionsClient(api_url, self.token)

    def login(self, api_url: str, username: str, password: str) -> str:
        try:
            response = login_to_api(api_url, username, password)
            token = response.get("access_token")
            return token
        except Exception as e:
            logging.error(e)
            raise e

    def health_check(self):
        return get_health_check(self.api_url, self.token)


def main():
    username = os.getenv("ERGODIC_API_USERNAME")
    password = os.getenv("ERGODIC_API_PASSWORD")
    api_url = "http://localhost:8000"

    if not username or not password:
        raise Exception("ERGODIC_API_USERNAME or ERGODIC_API_PASSWORD not set")

    client = ErgodicClient(api_url, username, password)
    print(client.health_check())


if __name__ == "__main__":
    main()
