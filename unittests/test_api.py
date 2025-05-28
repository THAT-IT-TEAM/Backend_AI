"""
Tests for the API endpoints of the AI chatbot application.
"""
import os
import json
import requests
from dotenv import load_dotenv
import unittest
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from the root directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Get the base URL from environment variable
BASE_URL = os.getenv('API_URL')
if not BASE_URL:
    raise EnvironmentError("API_URL environment variable is required. Please set it in your .env file.")

# Required environment variables
REQUIRED_ENV_VARS = [
    'GROQ_API_KEY',
    'SUPABASE_URL',
    'SUPABASE_KEY',
    'API_URL'  # Added API_URL as required
]

# Optional environment variables
OPTIONAL_ENV_VARS = [
    'TESSERACT_CMD'
]

class TestAPIEndpoints(unittest.TestCase):
    """Test cases for the API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up test class - runs once before all tests."""
        # Check for required environment variables
        missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                "Please ensure these are set in your .env file."
            )

        # Check for optional environment variables
        missing_optional = [var for var in OPTIONAL_ENV_VARS if not os.getenv(var)]
        if missing_optional:
            logger.warning(
                f"Optional environment variables not set: {', '.join(missing_optional)}. "
                "Some features may be limited."
            )

        cls.base_url = BASE_URL
        # Verify the server is running
        try:
            response = requests.get(f"{cls.base_url}/")
            if response.status_code != 404:  # We expect 404 as root endpoint doesn't exist
                raise ConnectionError("Server is not running or is not accessible")
            logger.info(f"Successfully connected to server at {cls.base_url}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to server at {cls.base_url}: {str(e)}")
            raise ConnectionError("Could not connect to the server. Make sure it's running.")

    def setUp(self):
        """Set up test data for each test method."""
        # Test document data for chat
        self.test_doc = {
            "document_id": "report.txt",
            "bucket_name": "data-storage",
            "question": "What is the main topic of this document?"
        }
        # Test OCR data with actual image URL
        self.test_ocr_url = "https://ykkxvbkcaiuajmeecmxe.supabase.co/storage/v1/object/public/images//Screenshot%202025-05-28%20000759.png"

    def _make_request(self, method, url, **kwargs):
        """Helper method to make requests with proper error handling and logging"""
        try:
            logger.info(f"Making {method} request to {url}")
            logger.debug(f"Request data: {json.dumps(kwargs.get('json', {}), indent=2)}")
            
            response = requests.request(method, url, **kwargs)
            
            logger.info(f"Response status code: {response.status_code}")
            try:
                response_data = response.json()
                logger.debug(f"Response data: {json.dumps(response_data, indent=2)}")
                if response.status_code >= 400:
                    logger.error(f"Error response: {response_data.get('error', 'Unknown error')}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse response as JSON: {response.text}")
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise

    def test_ocr_processing(self):
        """Test if OCR endpoint correctly processes a document and extracts data."""
        url = f"{self.base_url}/ocr"
        headers = {"Content-Type": "application/json"}
        
        test_data = {
            "file_url": self.test_ocr_url,
            "user_id": "92f61135-ad65-4e06-bf43-228be7f2d119",
            "trip_id": "6ac69507-b551-4ff1-8d09-6d11b27432d4"
        }
        
        response = self._make_request('POST', url, json=test_data, headers=headers)
        
        # Verify successful response
        self.assertEqual(response.status_code, 200, 
            f"OCR request failed with status {response.status_code}. "
            f"Error: {response.json().get('error', 'Unknown error') if response.status_code >= 400 else 'None'}")
        
        data = response.json()
        
        # Verify OCR response structure
        self.assertIn('extracted_data', data)
        self.assertIn('summary', data)
        self.assertIn('expense_id', data)
        self.assertIn('user_id', data)
        self.assertIn('trip_id', data)
        
        # Verify extracted data has required fields
        extracted_data = data['extracted_data']
        self.assertIn('Date', extracted_data)
        self.assertIn('Amount', extracted_data)
        self.assertIn('Vendor/Store', extracted_data)
        self.assertIn('Category', extracted_data)
        self.assertIn('Description', extracted_data)
        self.assertIn('Currency', extracted_data)
        self.assertIn('Tax Amount', extracted_data)
        
        # Verify data types
        self.assertIsInstance(extracted_data['Date'], str)
        self.assertIsInstance(extracted_data['Amount'], str)
        self.assertIsInstance(extracted_data['Vendor/Store'], str)
        self.assertIsInstance(extracted_data['Category'], str)
        self.assertIsInstance(extracted_data['Description'], str)
        self.assertIsInstance(extracted_data['Currency'], str)
        self.assertIsInstance(extracted_data['Tax Amount'], str)
        self.assertIsInstance(data['summary'], str)
        self.assertIsInstance(data['expense_id'], str)
        self.assertIsInstance(data['user_id'], str)
        self.assertIsInstance(data['trip_id'], str)
        
        # Verify non-empty values
        self.assertTrue(len(extracted_data['Date']) > 0)
        self.assertTrue(len(extracted_data['Amount']) > 0)
        self.assertTrue(len(extracted_data['Vendor/Store']) > 0)
        self.assertTrue(len(extracted_data['Category']) > 0)
        self.assertTrue(len(extracted_data['Description']) > 0)
        self.assertTrue(len(extracted_data['Currency']) > 0)
        self.assertTrue(len(data['summary']) > 0)
        self.assertTrue(len(data['expense_id']) > 0)
        self.assertTrue(len(data['user_id']) > 0)
        self.assertTrue(len(data['trip_id']) > 0)
        
        # Verify IDs match
        self.assertEqual(data['user_id'], test_data['user_id'])
        self.assertEqual(data['trip_id'], test_data['trip_id'])

    def test_chat_with_document(self):
        """Test if chat endpoint correctly answers questions about a document."""
        url = f"{self.base_url}/chat"
        headers = {"Content-Type": "application/json"}
        
        # First request creates the vector DB and answers the question
        response = requests.post(url, json=self.test_doc, headers=headers)
        
        # Verify successful response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertIn('response', data)
        self.assertIn('vector_db_name_used', data)
        
        # Verify response content
        self.assertIsInstance(data['response'], str)
        self.assertTrue(len(data['response']) > 0)
        self.assertIsInstance(data['vector_db_name_used'], str)
        self.assertTrue(len(data['vector_db_name_used']) > 0)
        
        # Store the vector DB name for follow-up question
        vector_db_name = data['vector_db_name_used']
        
        # Ask a follow-up question using the same vector DB
        follow_up_data = {
            "vector_db_name": vector_db_name,
            "question": "Can you provide more details about the main topic?"
        }
        
        response = requests.post(url, json=follow_up_data, headers=headers)
        
        # Verify successful response to follow-up question
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure and content
        self.assertIn('response', data)
        self.assertIsInstance(data['response'], str)
        self.assertTrue(len(data['response']) > 0)

    def test_receipt_fraud_detection(self):
        """Test if fraud detection endpoint correctly analyzes a receipt for potential fraud."""
        # First process the receipt through OCR to get an expense_id
        ocr_url = f"{self.base_url}/ocr"
        headers = {"Content-Type": "application/json"}
        
        ocr_data = {
            "file_url": self.test_ocr_url,
            "user_id": "92f61135-ad65-4e06-bf43-228be7f2d119",
            "trip_id": "6ac69507-b551-4ff1-8d09-6d11b27432d4"
        }
        
        # Get expense_id from OCR
        ocr_response = self._make_request('POST', ocr_url, json=ocr_data, headers=headers)
        self.assertEqual(ocr_response.status_code, 200,
            f"OCR request failed with status {ocr_response.status_code}. "
            f"Error: {ocr_response.json().get('error', 'Unknown error') if ocr_response.status_code >= 400 else 'None'}")
        
        ocr_result = ocr_response.json()
        expense_id = ocr_result['expense_id']
        
        # Now test fraud detection
        fraud_url = f"{self.base_url}/fraud-check"
        fraud_data = {
            "expense_id": expense_id,
            "file_url": self.test_ocr_url
        }
        
        response = self._make_request('POST', fraud_url, json=fraud_data, headers=headers)
        
        # Verify successful response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify fraud check response structure
        self.assertIn('fraud_check_id', data)
        self.assertIn('overall_risk_score', data)
        self.assertIn('fraud_probability', data)
        self.assertIn('risk_factors', data)
        self.assertIn('summary', data)
        
        # Verify data types
        self.assertIsInstance(data['fraud_check_id'], str)
        self.assertIsInstance(data['overall_risk_score'], (int, float))
        self.assertIsInstance(data['fraud_probability'], (int, float))
        self.assertIsInstance(data['risk_factors'], list)
        self.assertIsInstance(data['summary'], str)
        
        # Verify value ranges
        self.assertTrue(0 <= data['overall_risk_score'] <= 1)
        self.assertTrue(0 <= data['fraud_probability'] <= 1)
        
        # Verify risk factors
        self.assertTrue(len(data['risk_factors']) > 0)
        for factor in data['risk_factors']:
            self.assertIsInstance(factor, str)
            self.assertTrue(len(factor) > 0)
        
        # Verify summary content
        self.assertTrue(len(data['summary']) > 0)
        self.assertIn(str(int(data['fraud_probability'] * 100)), data['summary'])  # Summary should mention the probability
        
        # Test error cases
        # Test with invalid expense_id
        invalid_data = {
            "expense_id": "invalid-expense-id",
            "file_url": self.test_ocr_url
        }
        error_response = requests.post(fraud_url, json=invalid_data, headers=headers)
        self.assertEqual(error_response.status_code, 400)
        
        # Test with missing expense_id
        missing_data = {
            "file_url": self.test_ocr_url
        }
        error_response = requests.post(fraud_url, json=missing_data, headers=headers)
        self.assertEqual(error_response.status_code, 400)
        
        # Test with missing file_url
        missing_url_data = {
            "expense_id": expense_id
        }
        error_response = requests.post(fraud_url, json=missing_url_data, headers=headers)
        self.assertEqual(error_response.status_code, 400)

if __name__ == '__main__':
    unittest.main() 