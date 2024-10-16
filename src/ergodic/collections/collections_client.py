from ergodic.collections.utils import create_collection, create_collection_data
from typing import List


class CollectionsClient:
    def __init__(self, api_url: str, api_token: str):
        self.api_url = api_url
        self.api_token = api_token

    def create_collection(self, name: str, description: str, uuids: List[str]):
        return create_collection(self.api_url, self.api_token, uuids, name, description)

    def create_collection_data(
        self, collection_id: str, features: List[str], name: str, description: str
    ):
        return create_collection_data(
            self.api_url, self.api_token, collection_id, features, name, description
        )

    # def get_collection_data(self, collection_id: str):
    #     return get_collection_data(self.api_url, self.api_token, collection_id)
