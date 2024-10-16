from typing import List
from tqdm.auto import tqdm
from ergodic.client import ErgodicClient
import logging

logger = logging.getLogger(__name__)


class KeywordMatcher:
    def __init__(self, client: ErgodicClient):
        self.client = client
        self.data = {}
        self.matches = {}

    def load(self, asset_ids: List[str]):
        for asset_id in tqdm(asset_ids, desc="Loading data"):
            self.data[asset_id] = self.client.assets.get_data(asset_id)

    def match(self, keyword: str):
        self.matches[keyword] = {}
        for asset_id in tqdm(self.data, desc="Matching keywords"):
            self.matches[keyword][asset_id] = keyword in self.data[asset_id]

        return [
            self.client.assets.get(x)
            for x in tqdm(self.matches[keyword], desc="Getting asset info")
            if self.matches[keyword][x]
        ]
