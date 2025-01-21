from typing import List

from pydantic import BaseModel
from ergodic.tasks.extract import (
    fetch_status,
    request_features,
    TextDataRequest,
    fetch_final_data,
)


class ExtractionResponse:
    def __init__(self, info: dict, data_model: dict):
        self.info = info
        self.data_model = data_model
        self.data = None

    def wait(self):
        data = fetch_final_data(self.info["statusQueryGetUri"], wait=True)
        self.data = data
        return data

    def get_data_model(self):
        return self.data_model

    def get_status(self):
        return fetch_status(self.info["statusQueryGetUri"])

    def __repr__(self):
        status = self.get_status()
        return f"Extracting data... {status}"


class ExtractionClient:
    def __init__(self, api_url: str, api_token: str):
        self.api_url = api_url
        self.api_token = api_token

    def extract_features(
        self,
        data: List[dict | TextDataRequest],
        description: str | None = None,
        data_model: type[BaseModel] | None = None,
    ):
        response, generated_model = request_features(data, description, data_model)
        return ExtractionResponse(response, generated_model)
