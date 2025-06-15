import os
import requests
from dotenv import load_dotenv
from typing import Optional, Any
import asyncio # Needed for async def _get_service_token

load_dotenv()

# API_BASE_URL for the local backend
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:3050")

_ai_service_token = None

async def _get_service_token() -> Optional[str]:
    global _ai_service_token
    if _ai_service_token:
        print("DEBUG: Reusing existing AI service token.")
        return _ai_service_token

    print("DEBUG: Attempting to obtain AI service token...")
    # First, log in as admin to get an admin JWT
    admin_email = os.environ.get("ADMIN_EMAIL", "admin@blockchain.com")
    admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
    admin_wallet_id = os.environ.get("ADMIN_WALLET_ID", "0x0000000000000000000000000000000000000000") # This might need to be dynamically fetched or provided.

    try:
        print(f"DEBUG: Logging in as admin: {admin_email}")
        login_response = requests.post(f"{API_BASE_URL}/auth/login", json={
            "email": admin_email,
            "password": admin_password,
            "walletId": admin_wallet_id # Assuming this is required for admin login
        })
        login_response.raise_for_status()
        admin_token = login_response.json().get("token")

        if not admin_token:
            print("ERROR: Admin login successful but no token received.")
            return None
        print("DEBUG: Admin login successful, obtaining service token...")

        # Use admin token to get AI Service Token
        service_token_response = requests.post(
            f"{API_BASE_URL}/auth/service-token",
            headers={'Authorization': f'Bearer {admin_token}'}
        )
        service_token_response.raise_for_status()
        _ai_service_token = service_token_response.json().get("token")

        if _ai_service_token:
            print("DEBUG: Successfully obtained AI service token.")
            return _ai_service_token
        else:
            print("ERROR: Failed to obtain AI service token.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to obtain AI service token: {e}")
        print(f"Response body (if available): {e.response.text if e.response else 'N/A'}")
        return None

async def _authorized_request(method: str, url: str, **kwargs) -> requests.Response:
    token = await _get_service_token()
    if not token:
        raise Exception("AI Service Token not available.")

    headers = kwargs.get('headers', {})
    headers['Authorization'] = f'Bearer {token}'
    kwargs['headers'] = headers

    print(f"DEBUG: Making authorized {method.upper()} request to {url}")
    # Note: requests.request is synchronous. For true async, you'd use aiohttp or httpx.
    # However, since this is being called by an async function, asyncio will manage its execution.
    # This might block the event loop if the request is long-running.
    return requests.request(method, url, **kwargs) 