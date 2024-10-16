import requests
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()


def login_to_api(api_url: str, username: str, password: str):
    """
    Function to send a request to the FastAPI /token endpoint for user login.

    :param api_url: The base URL of your FastAPI server.
    :param username: The username (email) to log in with.
    :param password: The password for the provided username.
    :return: The JSON response containing the access token or error message.
    """
    # URL for the /token route
    token_url = f"{api_url}/token"

    # Data to be sent in the form-data format expected by OAuth2PasswordRequestForm
    data = {"username": username, "password": password}

    try:
        logging.info(f"Sending request to {token_url}")
        # Sending a POST request to the /token endpoint
        response = requests.post(token_url, data=data)

        # Check if the request was successful
        if response.status_code == 200:
            # Returning the token or the response data
            logging.info("Login successful")
            return response.json()
        else:
            # Handling error responses
            logging.error(response.json())
            raise Exception(response.json())

    except Exception as e:
        logging.error(e)
        raise e


def get_header(token: str) -> dict:
    """This function returns the headers to be used in the requests to the API

    :param token: The access token to be used in the Authorization header
    :return: The headers to be used in the requests
    """
    headers = {}
    headers["Authorization"] = f"Bearer {token}"
    return headers


def get_health_check(api_url: str, token: str) -> bool:
    """This function returns the health check of the API

    :param api_url: The base URL of your FastAPI server.
    :param token: The access token to be used in the Authorization header
    :return: The health check of the API
    """
    try:
        response = requests.get(f"{api_url}/health", headers=get_header(token))
        status = response.json().get("status") == "ok"

        if not status:
            raise Exception(response.json())

        return True
    except Exception as e:
        logging.error(e)
        return False


def main():
    print("hhere")
    username = os.getenv("ERGODIC_API_USERNAME")
    password = os.getenv("ERGODIC_API_PASSWORD")

    if not username or not password:
        raise Exception("ERGODIC_API_USERNAME or ERGODIC_API_PASSWORD not set")

    api_url = "http://localhost:8000"

    token = login_to_api(api_url, username, password)
    print(token)
    print(get_health_check(api_url, token["access_token"]))


if __name__ == "__main__":
    main()
