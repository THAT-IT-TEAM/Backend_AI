import os
import requests
from dotenv import load_dotenv
from auth_api_client import _authorized_request, API_BASE_URL # Import from the new auth_api_client

# Load environment variables from .env file in the current directory (ai folder)
load_dotenv()

# Remove AI_SERVICE_JWT as it's handled by _get_service_token in auth_api_client
# AI_SERVICE_JWT = os.environ.get("AI_SERVICE_JWT")

# if not AI_SERVICE_JWT:
#     print("Warning: AI_SERVICE_JWT not found in .env. API calls may fail.")

def get_auth_header() -> dict:
    """
    Returns the authorization header for API requests.
    (This will be less needed after migrating to _authorized_request for direct calls)
    """
    # This function is now deprecated as _authorized_request handles auth
    # It's kept for compatibility if other parts of the code still use it
    # but should ideally be refactored away.
    return {}

async def fetch_document_from_api(bucket_name: str, file_path: str) -> bytes | None:
    """
    Fetches a document from the Node.js API's static file serving endpoint asynchronously.

    Args:
        bucket_name: The name of the storage bucket.
        file_path: The path to the file within the bucket.

    Returns:
        The document content as bytes, or None if fetching fails.
    """
    try:
        file_url = f"{API_BASE_URL}/uploads/{bucket_name}/{file_path}"
        print(f"Fetching document from API: {file_url}")
        
        # Use the async authorized request helper
        response = await _authorized_request('GET', file_url, stream=True)
        response.raise_for_status() # Raise an exception for HTTP errors

        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching document from API: {e}")
        return None

async def delete_vector_db_document_via_api(document_id: str) -> bool:
    """
    Deletes a vector DB document record via the Node.js API asynchronously.

    Args:
        document_id: The ID of the document to delete.

    Returns:
        True if deletion was successful, False otherwise.
    """
    try:
        delete_url = f"{API_BASE_URL}/api/vector_db_documents/{document_id}"
        print(f"Deleting vector DB document via API: {delete_url}")
        
        # Use the async authorized request helper
        response = await _authorized_request('DELETE', delete_url)
        response.raise_for_status() # Raise an exception for HTTP errors
        print(f"Successfully deleted vector DB document {document_id} via API. Response: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error deleting vector DB document via API: {e}")
        return False

# Placeholder for generic API calls if needed later
async def get_records_from_api(table_name: str, query_params: dict = None) -> list | None:
    """
    Fetches records from a specified table via the Node.js API asynchronously.
    """
    try:
        url = f"{API_BASE_URL}/api/{table_name}"
        print(f"Fetching records from API: {url}")
        response = await _authorized_request('GET', url, params=query_params)
        response.raise_for_status()
        return response.json().get(table_name) # Assuming API returns { table_name: [...] }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching records from API: {e}")
        return None

async def create_record_via_api(table_name: str, data: dict) -> dict | None:
    """
    Creates a new record in a specified table via the Node.js API asynchronously.
    """
    try:
        url = f"{API_BASE_URL}/api/{table_name}"
        print(f"Creating record in {table_name} via API: {url}")
        response = await _authorized_request('POST', url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error creating record in {table_name} via API: {e}")
        return None 