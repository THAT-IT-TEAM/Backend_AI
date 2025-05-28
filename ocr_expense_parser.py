import os
from dotenv import load_dotenv
from groq import Groq
import json
from supabase import create_client, Client
from datetime import datetime
from typing import Optional, Dict, Any
import re
from uuid import UUID

load_dotenv()

# Initialize Groq client
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    print("Warning: GROQ_API_KEY not set in .env. Expense parsing will not work.")
    client = None
else:
    client = Groq(api_key=groq_api_key)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    print("Warning: SUPABASE_URL or SUPABASE_KEY not set in .env. Database storage will not work.")
    supabase: Optional[Client] = None
else:
    supabase = create_client(supabase_url, supabase_key)

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

def store_expense_data(parsed_data: Dict[str, Any], file_url: str, user_id: UUID, trip_id: UUID) -> Optional[str]:
    """
    Store the parsed expense data in Supabase.
    
    Args:
        parsed_data: The parsed expense data dictionary
        file_url: The URL of the original document
        user_id: UUID of the user who submitted the expense
        trip_id: UUID of the trip this expense belongs to
        
    Returns:
        The ID of the created expense record if successful, None otherwise
    """
    if supabase is None:
        print("Supabase client not initialized. Cannot store expense data.")
        return None
        
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
        
        # Prepare the data for insertion
        expense_data = {
            'user_id': str(user_id),  # Convert UUID to string for storage
            'trip_id': str(trip_id),  # Convert UUID to string for storage
            'amount': amount,
            'currency': currency,  # Use formatted currency
            'transaction_date': extracted_data.get('Date'),
            'vendor_name': extracted_data.get('Vendor/Store'),
            'category': extracted_data.get('Category'),
            'description': extracted_data.get('Description'),
            'document_id': extracted_data.get('Document ID or Reference Number'),
            'payment_method': extracted_data.get('Payment Method'),
            'tax_amount': tax_amount,
            'document_url': file_url,
            'extracted_data': extracted_data,
            'summary': parsed_data.get('summary')
        }
        print(f"Prepared expense data: {json.dumps(expense_data, indent=2)}")
        
        # Insert the data into Supabase
        print("Attempting to insert data into Supabase...")
        result = supabase.table('expenses').insert(expense_data).execute()
        print(f"Supabase insert result: {json.dumps(result.data if result.data else {}, indent=2)}")
        
        if result.data:
            print(f"Successfully stored expense with ID: {result.data[0]['id']}")
            return result.data[0]['id']
        print("No data returned from Supabase insert")
        return None
        
    except Exception as e:
        print(f"Error storing expense data in Supabase: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def parse_expense_text(file_url: str, user_id: UUID, trip_id: UUID) -> dict:
    """
    Uses Groq to parse an expense document and extract key details.
    Handles document URLs directly and stores the data in Supabase.

    Args:
        file_url: The URL of the expense document to analyze
        user_id: UUID of the user who submitted the expense
        trip_id: UUID of the trip this expense belongs to

    Returns:
        A dictionary containing:
        - expense_id: The Supabase record ID where the data is stored (None if storage failed)
        - user_id: The UUID of the user who submitted the expense
        - trip_id: The UUID of the trip this expense belongs to
        - extracted_data: The parsed expense details
        - summary: A summary of the extracted information
        - document_url: The URL of the original document
        - stored_at: Timestamp of when the data was processed
        Returns an empty dictionary if parsing fails or Groq client is not available.
    """
    if client is None:
        print("Groq client not initialized due to missing API key. Cannot parse expense.")
        return {}

    try:
        print(f"Starting expense parsing for file: {file_url}")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI assistant specialized in parsing expense documents.
                    Analyze the provided document and extract the following key details as a JSON object:
                    - Amount: Total amount paid (float or string, include currency if possible)
                    - Date: Transaction or invoice date (in YYYY-MM-DD format if possible)
                    - Vendor/Store: Name of the business or vendor
                    - Category: Type of expense (e.g., meals, travel, accommodation, supplies)
                    - Description: Brief note or itemized list if available
                    - Document ID or Reference Number: Any identifiable document number
                    - Currency: The currency of the amount
                    - Payment Method: Method used for payment (e.g., Credit Card, Cash)
                    - Tax Amount: If present and relevant (float or string, include currency if possible)
                    
                    Also provide a concise summary of the key information extracted.
                    
                    Format your response as a JSON object with two top-level keys:
                    - 'extracted_data': containing the JSON object of extracted details
                    - 'summary': containing the summary string
                    
                    If a piece of information is not found, use null for that key in the JSON object.""",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this expense document and extract the key details."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": file_url
                            }
                        }
                    ]
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        response_content = chat_completion.choices[0].message.content
        print(f"Raw Groq response: {response_content}")
        
        try:
            parsed_response = json.loads(response_content)
            print(f"Parsed Groq response: {json.dumps(parsed_response, indent=2)}")
            
            # Always return a consistent response structure
            response = {
                "expense_id": None,  # Will be updated if storage succeeds
                "user_id": str(user_id),
                "trip_id": str(trip_id),
                "extracted_data": parsed_response.get("extracted_data", {}),
                "summary": parsed_response.get("summary", ""),
                "document_url": file_url,
                "stored_at": datetime.utcnow().isoformat()
            }
            
            # Store the parsed data in Supabase
            if parsed_response:
                print("Attempting to store parsed data in Supabase...")
                expense_id = store_expense_data(parsed_response, file_url, user_id, trip_id)
                if expense_id:
                    print(f"Successfully stored expense with ID: {expense_id}")
                    response["expense_id"] = expense_id
                else:
                    print("Warning: Failed to store data in Supabase")
            
            return response
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Groq response: {e}")
            print(f"Raw Groq response content: {response_content}")
            return {}

    except Exception as e:
        print(f"Error interacting with Groq API for expense parsing: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return {}

# Example Usage
# if __name__ == "__main__":
#     from uuid import uuid4
#     sample_url = "https://example.com/path/to/receipt.jpg"
#     user_id = uuid4()  # Replace with actual user UUID
#     trip_id = uuid4()  # Replace with actual trip UUID
#     parsed_data = parse_expense_text(sample_url, user_id, trip_id)
#     if parsed_data:
#         print("Parsed Expense Data:")
#         print(f"Expense ID: {parsed_data.get('expense_id', 'Not stored')}")
#         print(f"User ID: {parsed_data.get('user_id')}")
#         print(f"Trip ID: {parsed_data.get('trip_id')}")
#         print("Extracted Data:")
#         print(json.dumps(parsed_data.get('extracted_data', {}), indent=2))
#         print("\nSummary:")
#         print(parsed_data.get('summary', ''))
#     else:
#         print("Failed to parse expense data.") 