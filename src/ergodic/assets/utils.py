import os
from typing import Optional
import requests
import logging
from ergodic.auth import get_header

logger = logging.getLogger(__name__)


def list_assets(api_url: str, api_token: str):
    logger.info("Listing assets")
    try:
        headers = get_header(api_token)
        response = requests.get(f"{api_url}/assets/list", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error listing assets: {str(e)}")
        raise e


def get_asset(api_url: str, api_token: str, asset_id: str):
    logger.info(f"Getting asset with ID: {asset_id}")
    try:
        headers = get_header(api_token)
        response = requests.get(f"{api_url}/assets/get/{asset_id}", headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error getting asset {asset_id}: {str(e)}")
        raise e


def upload_pdf(
    api_url: str,
    api_token: str,
    pdf_path: str,
    name: Optional[str] = None,
    context: Optional[str] = None,
):
    logger.info(f"Uploading PDF: {pdf_path}")
    try:
        headers = get_header(api_token)
        with open(pdf_path, "rb") as file:
            files = {"file": (pdf_path, file, "application/pdf")}
            response = requests.post(
                f"{api_url}/upload_any", headers=headers, files=files
            )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, IOError) as e:
        logger.error(f"Error uploading PDF {pdf_path}: {str(e)}")
        raise e


def upload_pdfs_from_folder(
    api_url: str, api_token: str, folder_path: str, context: Optional[str] = None
):
    logger.info(f"Uploading PDFs from folder: {folder_path}")
    results = []
    try:
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(folder_path, file)
                results.append(
                    upload_pdf(api_url, api_token, pdf_path, context=context)
                )
        logger.info("Finished uploading PDFs from folder")
        return results
    except Exception as e:
        logger.error(f"Error uploading PDFs from folder {folder_path}: {str(e)}")
        raise e


def get_asset_data(api_url: str, api_token: str, asset_id: str, n_max: int = 1000):
    """
    Get the data from an asset.
    """
    logger.info(f"Getting data from asset {asset_id}")
    try:
        headers = get_header(api_token)
        data = {"asset_id": asset_id, "n_max": n_max}
        response = requests.post(f"{api_url}/assets/data", headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error getting asset {asset_id}: {str(e)}")
        raise e
