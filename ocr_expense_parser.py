import os
from dotenv import load_dotenv
from groq import Groq
import json

load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    # In a real application, you might handle this more gracefully than raising an error here
    print("Warning: GROQ_API_KEY not set in .env. Expense parsing will not work.")
    client = None # Set client to None if API key is missing
else:
    client = Groq(api_key=groq_api_key)

def parse_expense_text(file_url: str) -> dict:
    """
    Uses Groq to parse an expense document and extract key details.
    Handles document URLs directly.

    Args:
        file_url: The URL of the expense document to analyze.

    Returns:
        A dictionary containing the extracted expense details and a summary.
        Returns an empty dictionary if parsing fails or Groq client is not available.
    """
    if client is None:
        print("Groq client not initialized due to missing API key. Cannot parse expense.")
        return {}

    try:
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
        try:
            parsed_response = json.loads(response_content)
            return parsed_response
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Groq response: {e}")
            print(f"Raw Groq response content: {response_content}")
            return {}

    except Exception as e:
        print(f"Error interacting with Groq API for expense parsing: {e}")
        return {}

# Example Usage
# if __name__ == "__main__":
#     sample_url = "https://example.com/path/to/receipt.jpg"
#     parsed_data = parse_expense_text(sample_url)
#     if parsed_data:
#         print("Parsed Expense Data:")
#         print(json.dumps(parsed_data, indent=2))
#     else:
#         print("Failed to parse expense data.") 