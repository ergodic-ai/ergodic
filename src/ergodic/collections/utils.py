from typing import List
import logging
import requests
from ergodic.auth import get_header

logger = logging.getLogger(__name__)


def create_collection(
    api_url: str, api_token: str, uuids: List[str], name: str, description: str
):
    """
    Create a new collection using the provided API.

    Args:
        api_url (str): The base URL of the API.
        api_token (str): The authentication token for the API.
        uuids (List[str]): A list of PDF UUIDs to include in the collection.
        name (str): The name of the collection.
        description (str): A description of the collection.

    Returns:
        dict: The JSON response from the API containing the created collection details.

    Raises:
        requests.RequestException: If there's an error during the API request.
    """
    headers = get_header(api_token)
    logger.info(f"Creating collection with name: {name}")
    try:
        response = requests.post(
            f"{api_url}/create_collection",
            headers=headers,
            json={"name": name, "description": description, "pdf_uuids": uuids},
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise


def create_collection_data(
    api_url: str,
    api_token: str,
    collection_id: str,
    features: List[str],
    name: str,
    description: str,
):
    """
    Create collection data for an existing collection using the provided API.

    Args:
        api_url (str): The base URL of the API.
        api_token (str): The authentication token for the API.
        collection_id (str): The UUID of the collection to add data to.
        features (List[str]): A list of features to include in the collection data.
        name (str): The name of the collection data.
        description (str): A description of the collection data.

    Returns:
        dict: The JSON response from the API containing the created collection data details.

    Raises:
        requests.RequestException: If there's an error during the API request.
    """
    headers = get_header(api_token)
    logger.info(f"Creating collection data with name: {name}")
    try:
        response = requests.post(
            f"{api_url}/create_collection_data",
            headers=headers,
            json={
                "selected_features": features,
                "name": name,
                "description": description,
                "collection_uuid": collection_id,
            },
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error creating collection data: {str(e)}")
        raise
