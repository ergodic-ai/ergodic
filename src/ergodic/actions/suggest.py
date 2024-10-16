# @router.get("/features/suggestions/{asset_id}")
# async def suggest_features(
#     asset_id: str,
#     current_user: UserFrontEnd = Depends(get_current_user),
#     db_client: AssetDBClientBase = Depends(get_assetdb_client),
# ):
#     features = await suggest_features_for_single_asset(asset_id, db_client)
#     return features

import requests
from ergodic.auth import get_header
from ergodic.client import ErgodicClient


def _ask_for_feature_suggestions(api_url: str, api_token: str, asset_id: str):
    headers = get_header(api_token)
    response = requests.get(
        f"{api_url}/features/suggestions/{asset_id}", headers=headers
    )
    response.raise_for_status()
    return response.json()


def features(client: ErgodicClient, asset_id: str):
    return _ask_for_feature_suggestions(client.api_url, client.token, asset_id)
