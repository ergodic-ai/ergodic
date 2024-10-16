from ergodic.assets.utils import (
    list_assets,
    get_asset,
    get_asset_data,
    upload_pdf,
    upload_pdfs_from_folder,
)
from typing import Optional


class AssetClient:
    def __init__(self, api_url: str, api_token: str):
        self.api_token = api_token
        self.api_url = api_url

    def list(self):
        return list_assets(self.api_url, self.api_token)

    def upload_pdf(
        self, pdf_path: str, name: Optional[str] = None, context: Optional[str] = None
    ):
        return upload_pdf(self.api_url, self.api_token, pdf_path, name, context)

    def upload_pdfs_from_folder(self, folder_path: str, context: Optional[str] = None):
        return upload_pdfs_from_folder(
            self.api_url, self.api_token, folder_path, context
        )

    def get(self, asset_id: str):
        return get_asset(self.api_url, self.api_token, asset_id)

    def get_data(self, asset_id: str, n_max: int = 1000):
        return get_asset_data(self.api_url, self.api_token, asset_id, n_max)


def main():
    import os

    from dotenv import load_dotenv

    load_dotenv()

    api_url = "http://localhost:8000"
    username = os.getenv("ERGODIC_API_USERNAME")
    password = os.getenv("ERGODIC_API_PASSWORD")

    if not username or not password:
        raise Exception("ERGODIC_API_USERNAME or ERGODIC_API_PASSWORD not set")

    from ergodic.client import ErgodicClient

    client = ErgodicClient(api_url, username, password)
    token = client.token

    print(token)

    asset_client = AssetClient(api_url, token)
    print(asset_client.list())


if __name__ == "__main__":
    main()
