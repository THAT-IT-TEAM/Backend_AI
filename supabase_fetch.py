import requests
import os

# Assuming the local API is running on localhost:3050
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:3050")

def fetch_document_from_local_api(bucket_name: str, file_path: str):
    """
    Fetches a document from the local API's file storage.

    Args:
        bucket_name: The name of the storage bucket (e.g., 'data-storage').
        file_path: The path to the file within the bucket.

    Returns:
        The document content as bytes, or None if fetching fails.
    """
    try:
        # Construct the URL for the local file server
        url = f"{API_BASE_URL}/uploads/{bucket_name}/{file_path}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching document from local API: {e}")
        return None

# Example usage (for testing, can be removed later)
if __name__ == "__main__":
    # Make sure your backend API is running for this example to work
    # This example assumes you have a file named 'test.txt' in the 'data-storage' bucket
    # You might need to manually upload a test file via the UI or directly to Backend_Blockchain/uploads/data-storage/
    bucket = "data-storage"
    # Replace with an actual file path that exists in your 'data-storage' bucket
    # For instance, if you uploaded a file named 'my_document.pdf'
    path = "example.txt" # You need to replace this with a real file that exists in your bucket
    
    print(f"Attempting to fetch document from {API_BASE_URL}/uploads/{bucket}/{path}")
    document_content = fetch_document_from_local_api(bucket, path)
    if document_content:
        print(f"Successfully fetched document of size: {len(document_content)} bytes")
        print("Content snippet:", document_content[:100].decode('utf-8')) # Print first 100 chars
    else:
        print("Failed to fetch document") 