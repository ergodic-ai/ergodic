from typing import List
import requests
import time
from pydantic import BaseModel
import logging
from openai.lib._pydantic import to_strict_json_schema

BASE_URL_REMOTE = "https://scraperfns.azurewebsites.net/api"


class TextDataRequest(BaseModel):
    text: str
    title: str


def make_request(
    endpoint: str,
    data: dict,
    code: str = "",
    method: str = "POST",
) -> dict:
    """Helper function to make requests to the API
    Args:
        endpoint: The endpoint to make the request to
        data: The data to send to the endpoint
        code: The code to use for the request
        method: The method to use for the request
    """
    base_url = BASE_URL_REMOTE
    url = f"{base_url}/{endpoint}"

    if code:
        url = f"{url}?code={code}"

    headers = {"Content-Type": "application/json"}

    if method == "GET":
        response = requests.get(url, params=data, headers=headers)
    else:
        response = requests.post(url, json=data, headers=headers)

    return response.json()


def fetch_status(statusQueryGetUri: str) -> dict:
    response = requests.get(statusQueryGetUri)
    return response.json()["runtimeStatus"]


def fetch_final_data(statusQueryGetUri: str, wait: bool = True) -> dict:
    response = requests.get(statusQueryGetUri)
    response_json = response.json()
    if response_json["runtimeStatus"] == "Completed":
        return response_json["output"]
    elif wait:
        logging.info("Waiting for extraction to complete...")
        # Show progress indicator
        print(".", end="", flush=True)
        time.sleep(30)
        return fetch_final_data(statusQueryGetUri, wait=True)
    else:
        return response_json


def get_schema(description: str) -> dict:
    endpoint = "json_schema"
    data = {"features": description}
    response = make_request(endpoint, data, method="POST")
    return response


def request_features(
    request_data: List[dict | TextDataRequest],
    description: str | None = None,
    data_model: type[BaseModel] | None = None,
) -> tuple[dict, dict]:
    assert len(request_data) > 0, "request_data must contain at least one item"
    assert isinstance(
        request_data[0], (dict, TextDataRequest)
    ), "request_data must contain only dict or TextDataRequest items"

    schema: dict = {}
    if description:
        logging.info(f"Getting schema for description: {description}")
        schema = get_schema(description)
        logging.info(f"Schema: {schema}")
    elif data_model:
        schema = {
            "type": "json_schema",
            "json_schema": {
                "schema": to_strict_json_schema(data_model),
                "name": data_model.__name__,
                "strict": True,
            },
        }
    else:
        raise ValueError("Either description or data_model must be provided")

    request_data_dict = []
    for element in request_data:
        if isinstance(element, TextDataRequest):
            d = {"text": element.text, "title": element.title}
        else:
            # validate that the element has the right keys
            assert (
                "text" in element and "title" in element
            ), "element must have text and title keys"
            d = element
        request_data_dict.append(d)

    logging.info(f"Request data: {request_data_dict}")

    endpoint = "orchestrators/extract_text"
    payload = {"schema": schema, "data": request_data_dict}
    status = make_request(endpoint, payload, method="POST")

    logging.info(f"Status: {status}")

    return status, schema
