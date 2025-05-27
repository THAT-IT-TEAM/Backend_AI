import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Supabase URL and Key must be set in .env")

supabase: Client = create_client(url, key)

def fetch_document_from_supabase(bucket_name: str, file_path: str):
    """
    Fetches a document from a Supabase storage bucket.

    Args:
        bucket_name: The name of the Supabase storage bucket.
        file_path: The path to the file within the bucket.

    Returns:
        The document content as bytes, or None if fetching fails.
    """
    try:
        # Assuming the document is stored in storage and you want to download it
        res = supabase.storage.from_(bucket_name).download(file_path)
        return res
    except Exception as e:
        print(f"Error fetching document from Supabase: {e}")
        return None

# Example usage (you can remove or comment this out later)
# if __name__ == "__main__":
#     # Replace with your bucket name and file path
#     bucket = "your_bucket_name"
#     path = "your_file_path"
#     document_content = fetch_document_from_supabase(bucket, path)
#     if document_content:
#         print(f"Successfully fetched document of size: {len(document_content)} bytes")
#     else:
#         print("Failed to fetch document") 