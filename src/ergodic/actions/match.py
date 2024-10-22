from typing import List, Literal
from tqdm.auto import tqdm
from ergodic.client import ErgodicClient
import logging

# import spacy

logger = logging.getLogger(__name__)


class KeywordMatcher:
    def __init__(
        self,
        client: ErgodicClient,
        match_type: Literal["fuzzy", "exact"] = "exact",
        fuzzy_match_threshold: float = 0.8,
    ):
        self.client = client
        self.data = {}
        self.matches = {}
        self.match_type = match_type
        self.fuzzy_match_threshold = fuzzy_match_threshold

    def load(self, asset_ids: List[str]):
        for asset_id in tqdm(asset_ids, desc="Loading data"):
            self.data[asset_id] = self.client.assets.get_data(asset_id)

    def _match_strict(self, keyword: str):
        self.matches[keyword] = {}
        for asset_id in tqdm(self.data, desc="Matching keywords"):
            self.matches[keyword][asset_id] = keyword in self.data[asset_id]

        return [
            self.client.assets.get(x)
            for x in tqdm(self.matches[keyword], desc="Getting asset info")
            if self.matches[keyword][x]
        ]

    def _match_fuzzy(self, keyword: str):
        pass

        # Load the English language model
        # nlp = spacy.load("en_core_web_sm")

        # self.matches[keyword] = {}
        # keyword_doc = nlp(keyword.lower())

        # for asset_id in tqdm(self.data, desc="Fuzzy matching keywords"):
        #     asset_doc = nlp(self.data[asset_id].lower())
        #     similarity = keyword_doc.similarity(asset_doc)
        #     self.matches[keyword][asset_id] = similarity >= self.fuzzy_match_threshold

        # return [
        #     self.client.assets.get(x)
        #     for x in tqdm(self.matches[keyword], desc="Getting asset info")
        #     if self.matches[keyword][x]
        # ]
        # pass

    def match(self, keyword: str):
        if self.match_type == "exact":
            return self._match_strict(keyword)
        elif self.match_type == "fuzzy":
            return self._match_fuzzy(keyword)
