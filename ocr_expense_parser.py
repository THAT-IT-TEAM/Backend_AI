import os
from dotenv import load_dotenv
from groq import Groq
import json
# from supabase import create_client, Client # Removed Supabase import
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import re
from uuid import UUID
import requests
from openai import OpenAI
from auth_api_client import _authorized_request, API_BASE_URL # Import from the new auth_api_client
# from api_client import fetch_document_from_api # Add this import - REMOVED
import uuid # Add this import

load_dotenv()

# Initialize Groq client
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("Warning: GROQ_API_KEY not set in .env. Expense parsing will not work.")
    client = None
else:
    client = Groq(api_key=groq_api_key)

# Supabase client is no longer directly initialized here
# supabase: Optional[Client] = None # Removed Supabase client initialization

# Remove the moved authentication related functions and global variable
# _ai_service_token = None
# async def _get_service_token() -> Optional[str]:
#     ...
# async def _authorized_request(method: str, url: str, **kwargs) -> requests.Response:
#     ...

def clean_numeric_value(value: Any) -> Optional[float]:
    """
    Clean a numeric value by removing currency symbols and converting to float.
    
    Args:
        value: The value to clean (can be string, float, or None)
        
    Returns:
        Cleaned float value or None if conversion fails
    """
    if value is None:
        return None
        
    if isinstance(value, (int, float)):
        return float(value)
        
    if isinstance(value, str):
        # Remove currency symbols and other non-numeric characters except decimal point
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned)
        except ValueError:
            return None
            
    return None

async def get_wallet_addresses(user_id: UUID, vendor_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get wallet addresses for both user and vendor from the local API's profiles table.
    
    Args:
        user_id: UUID of the user
        vendor_name: Name of the vendor
        
    Returns:
        Tuple of (user_wallet_address, vendor_wallet_address)
    """
    try:
        user_wallet = None
        vendor_wallet = None

        # Fetch user's wallet address from local API profiles
        user_profile_response = await _authorized_request('GET', f"{API_BASE_URL}/api/profiles", params={'user_id': str(user_id)})
        user_profile_response.raise_for_status()
        user_profiles = user_profile_response.json().get('profiles', [])
        if user_profiles:
            user_wallet = user_profiles[0].get('wallet_id')
            print(f"Fetched user wallet: {user_wallet}")

        # Fetch vendor's wallet address from local API profiles
        vendor_profile_response = await _authorized_request('GET', f"{API_BASE_URL}/api/profiles", params={'email': vendor_name, 'role': 'vendor'})
        vendor_profile_response.raise_for_status()
        vendor_profiles = vendor_profile_response.json().get('profiles', [])
        if vendor_profiles:
            vendor_wallet = vendor_profiles[0].get('wallet_id')
            print(f"Fetched vendor wallet: {vendor_wallet}")
        else: # Fallback: if vendor not found by email, try by name. This might require backend changes.
            print(f"Vendor profile not found by email {vendor_name}, attempting to search by name if backend supports it.")

        return user_wallet, vendor_wallet
    except requests.exceptions.RequestException as e:
        print(f"Error getting wallet addresses from local API: {e}")
        return None, None

async def send_to_blockchain(expense_id: str, user_wallet: str, vendor_wallet: str, amount: float, 
                      category: str, description: Optional[str] = None, receipt_hash: Optional[str] = None) -> bool:
    """
    Send expense data to blockchain via Node.js API.
    
    Args:
        expense_id: The ID of the expense record in our local database
        user_wallet: User's blockchain wallet address
        vendor_wallet: Vendor's blockchain wallet address
        amount: Expense amount
        category: Expense category
        description: Optional expense description
        receipt_hash: Optional receipt hash/document ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    blockchain_api_url = os.environ.get("BLOCKCHAIN_API_URL") or API_BASE_URL # Use API_BASE_URL as fallback
    if not blockchain_api_url:
        print("Warning: BLOCKCHAIN_API_URL or API_BASE_URL not set in .env. Blockchain integration will not work.")
        return False
        
    try:
        # Adjusted payload to match backend's expected `expenseData`
        payload = {
            "amount": amount,
            "currency": "USD", # Defaulting for now, could be extracted by OCR
            "transaction_date": datetime.now().isoformat(), # Default, should come from OCR
            "vendor_name": vendor_name,
            "category": category,
            "description": description,
            "document_id": receipt_hash,
            "user_id": "", # User ID will be taken from token on backend
            "trip_id": "", # Needs to be passed if available
            "payment_method": "", # Needs to be extracted/provided
            "tax_amount": 0, # Needs to be extracted/provided
            "document_url": "", # This will be the direct URL to the uploaded file
            "extracted_data": "", # Will be set by OCR parsing output
            "summary": "" # Will be set by LLM output
        }
        
        response = await _authorized_request('POST', f"{blockchain_api_url}/api/expenses", json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        # Backend should return the newly created expense ID and blockchain_id
        response_data = response.json()
        new_expense_id = response_data.get('expenseId')
        blockchain_id = response_data.get('transactionHash')
        
        print(f"Successfully created expense via API: {new_expense_id}, Blockchain ID: {blockchain_id}")
        return new_expense_id # Indicate success of API call
        
    except requests.exceptions.RequestException as e:
        print(f"Error creating expense via API: {e}")
        return False

async def store_expense_data(parsed_data: Dict[str, Any], file_url: str, user_id: UUID, trip_id: UUID) -> Optional[str]:
    """
    Store the parsed expense data in the local database via API.
    
    Args:
        parsed_data: The parsed expense data dictionary
        file_url: The URL of the original document
        user_id: UUID of the user who submitted the expense
        trip_id: UUID of the trip this expense belongs to
        
    Returns:
        The ID of the created expense record if successful, None otherwise
    """
    try:
        print(f"Attempting to store expense data for user {user_id} and trip {trip_id}")
        extracted_data = parsed_data.get('extracted_data', {})
        print(f"Extracted data: {json.dumps(extracted_data, indent=2)}")
        
        # Clean numeric values before storing
        amount = clean_numeric_value(extracted_data.get('Amount'))
        tax_amount = clean_numeric_value(extracted_data.get('Tax Amount'))
        print(f"Cleaned numeric values - amount: {amount}, tax_amount: {tax_amount}")
        
        # Format currency to fit database constraints (max 10 chars)
        currency = extracted_data.get('Currency', '')
        if currency:
            # Extract currency code or symbol if present
            if '(' in currency and ')' in currency:
                # Extract text between parentheses (usually the symbol)
                currency = currency[currency.find('(')+1:currency.find(')')].strip()
            # If still too long, take first 10 chars
            currency = currency[:10]
        
        vendor_name = extracted_data.get('Vendor/Store')
        category = extracted_data.get('Category')
        description = extracted_data.get('Description')
        document_id = extracted_data.get('Document ID or Reference Number')
        
        # Generate a UUID for the expense ID
        new_expense_uuid = str(uuid.uuid4())

        # Prepare the data for insertion to our local API
        # User ID and Trip ID are now expected as TEXT (UUID strings)
        expense_data = {
            'id': new_expense_uuid, # Include the generated UUID as the ID
            'user_id': str(user_id), 
            'trip_id': str(trip_id), 
            'amount': amount,
            'currency': currency,  
            'transaction_date': extracted_data.get('Date'),
            'vendor_name': vendor_name,
            'category': category,
            'description': description,
            'document_id': document_id,
            'payment_method': extracted_data.get('Payment Method'),
            'tax_amount': tax_amount,
            'document_url': file_url,
            'extracted_data': json.dumps(extracted_data), # Store as JSON string
            'summary': parsed_data.get('summary')
        }
        print(f"Prepared expense data for local API: {json.dumps(expense_data, indent=2)}")
        
        # Insert the data via our local API
        print("Attempting to insert data into local API...")
        response = await _authorized_request('POST', f"{API_BASE_URL}/api/expenses", json=expense_data)
        response.raise_for_status() # Raise exception for 4XX/5XX status codes
        
        result_data = response.json()
        print(f"DEBUG: Response from local API: {json.dumps(result_data, indent=2)}")
        print(f"DEBUG: result_data.get('expenseId'): {result_data.get('expenseId')}")

        if result_data and result_data.get('expenseId'):
            expense_id = result_data['expenseId']
            blockchain_id = result_data.get('transactionHash')
            print(f"Successfully stored expense with ID: {expense_id}, Blockchain ID: {blockchain_id}")
            
            return expense_id
            
        print("No data or ID returned from local API expense creation.")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error storing expense data via local API: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

async def parse_expense_text(file_url: str, user_id: UUID, trip_id: UUID) -> dict:
    """
    Uses Groq to parse an expense document and extract key details.
    Fetches document from local API and stores the data via local API.

    Args:
        file_url: The URL of the expense document (e.g., '/uploads/bucket/filename.pdf')
        user_id: UUID of the user who submitted the expense
        trip_id: UUID of the trip this expense belongs to

    Returns:
        A dictionary containing:
        - expense_id: The local database record ID where the data is stored (None if storage failed)
        - user_id: The UUID of the user who submitted the expense
        - trip_id: The UUID of the trip this expense belongs to
        - extracted_data: The parsed expense details
        - summary: A summary of the extracted information
        - document_url: The URL of the original document
        - stored_at: Timestamp of when the data was processed
    """
    print(f"Starting expense parsing for URL: {file_url}")

    # The document_content fetching is no longer needed as we are sending the image_url directly
    # path_parts = file_url.split('/uploads/')
    # if len(path_parts) < 2: # Check if '/uploads/' is in the URL
    #     print(f"Invalid file_url format: {file_url}")
    #     return {"expense_id": None, "user_id": user_id, "trip_id": trip_id, "extracted_data": {}, "summary": "", "document_url": file_url, "stored_at": datetime.now().isoformat()}
    # bucket_file_path = path_parts[1] # e.g., 'data-storage/my_receipt.pdf'
    # bucket_name = bucket_file_path.split('/')[0]
    # file_path_in_bucket = '/'.join(bucket_file_path.split('/')[1:])
    # document_content = await fetch_document_from_api(bucket_name, file_path_in_bucket)
    # if not document_content:
    #     print(f"Failed to fetch document content from local API for {file_url}")
    #     return {"expense_id": None, "user_id": user_id, "trip_id": trip_id, "extracted_data": {}, "summary": "", "document_url": file_url, "stored_at": datetime.now().isoformat()}
    # text_content = document_content.decode('utf-8', errors='ignore') # Assuming text content for LLM

    if client is None:
        print("Groq client not initialized. Cannot parse expense text.")
        return {"expense_id": None, "user_id": user_id, "trip_id": trip_id, "extracted_data": {}, "summary": "Groq client not available.", "document_url": file_url, "stored_at": datetime.now().isoformat()}

    try:
        # Define the system prompt content
        system_prompt_content = (
            "You are an expert expense parser. Extract the following details from the provided text:\n"
            "- Vendor/Store Name\n"
            "- Amount\n"
            "- Currency (e.g., USD, EUR)\n"
            "- Date (prefer YYYY-MM-DD or readable format)\n"
            "- Category (e.g., Food, Travel, Utilities, Office Supplies)\n"
            "- Description (brief summary of items/services)\n"
            "- Payment Method\n"
            "- Tax Amount (if present)\n"
            "- Document ID or Reference Number (e.g., Invoice #, Receipt #)\n"
            "Return the extracted data as a JSON object. If a field is not found, omit it. Also provide a one-sentence summary of the expense.\n"
            "Example JSON: {\"Vendor/Store\": \"Coffee Shop\", \"Amount\": 5.50, \"Currency\": \"USD\", \"Date\": \"2023-10-26\", \"Category\": \"Food\", \"Description\": \"Coffee and pastry\", \"Payment Method\": \"Credit Card\", \"Tax Amount\": 0.50, \"Document ID or Reference Number\": \"INV123\"}"
        )

        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct", # User specified model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt_content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": file_url # Use the file_url directly
                            }
                        }
                    ]
                }
            ],
            temperature=1, # User specified
            max_tokens=1024, # User specified max_completion_tokens (use max_tokens for Groq)
            top_p=1, # User specified
            stream=True, # User specified
            stop=None, # User specified
        )

        response_content = ""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="") # Print as it comes in for debugging/user feedback

        print(f"\nRaw Groq response (full): {response_content}") # Log full response after streaming

        # Attempt to parse the JSON and extract summary
        parsed_data = {}
        summary = ""
        try:
            # Groq might return text before/after JSON, or just JSON
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                parsed_data = json.loads(json_string)
                # Extract summary if it's part of the JSON, then remove it from extracted_data
                if "summary" in parsed_data:
                    summary = parsed_data["summary"]
                    del parsed_data["summary"]
            else:
                # If no JSON, treat whole response as a summary or error
                summary = response_content.strip()
                print("Warning: No JSON object found in Groq response.")

        except json.JSONDecodeError as e:
            print(f"JSON parsing error from Groq response: {e}")
            summary = f"Failed to parse Groq response: {response_content[:100]}... Error: {e}"

        # Store the expense data via our local API
        # The backend now handles blockchain integration on POST to /api/expenses
        expense_id = await store_expense_data(
            parsed_data={'extracted_data': parsed_data, 'summary': summary},
            file_url=file_url, 
            user_id=user_id, 
            trip_id=trip_id
        )

        return {
            "expense_id": expense_id,
            "user_id": user_id,
            "trip_id": trip_id,
            "extracted_data": parsed_data,
            "summary": summary,
            "document_url": file_url,
            "stored_at": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error during Groq parsing or data storage: {e}")
        return {"expense_id": None, "user_id": user_id, "trip_id": trip_id, "extracted_data": {}, "summary": f"Parsing failed: {e}", "document_url": file_url, "stored_at": datetime.now().isoformat()}

# Example Usage (for testing)
if __name__ == "__main__":
    import asyncio
    # Ensure your Node.js backend is running on localhost:3050
    # Ensure you have a GROQ_API_KEY and BLOCKCHAIN_API_URL set in your .env

    async def main_test():
        # Example document URL (replace with a real URL from your local API's /uploads)
        # You would typically get this URL after uploading a file via the frontend
        test_file_url = "http://localhost:3050/uploads/data-storage/sample_receipt.pdf"
        # Make sure you have a sample_receipt.pdf in Backend_Blockchain/uploads/data-storage/
        # or replace with a file path that actually exists after uploading it.

        # Dummy user and trip IDs (replace with actual UUIDs from your DB for testing)
        test_user_id = UUID('00000000-0000-0000-0000-000000000001') # Replace with a real user ID
        test_trip_id = UUID('00000000-0000-0000-0000-000000000002') # Replace with a real trip ID

        print(f"Testing OCR expense parsing for: {test_file_url}")
        parsed_result = await parse_expense_text(test_file_url, test_user_id, test_trip_id)
        print("\n--- Parsed Expense Result ---")
        print(json.dumps(parsed_result, indent=2)) 
    
    asyncio.run(main_test()) 