"""
Tests for the API endpoints of the AI chatbot application.
"""
import os
import json
import requests
from dotenv import load_dotenv
import unittest
from pathlib import Path

# Load environment variables from the root directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Get the base URL from environment variable or use the ngrok URL
BASE_URL = os.getenv('API_URL', 'https://platypus-valued-immensely.ngrok-free.app')

class TestAPIEndpoints(unittest.TestCase):
    """Test cases for the API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Set up test class - runs once before all tests."""
        cls.base_url = BASE_URL
        # Verify the server is running
        try:
            response = requests.get(f"{cls.base_url}/")
            if response.status_code != 404:  # We expect 404 as root endpoint doesn't exist
                raise ConnectionError("Server is not running or is not accessible")
        except requests.exceptions.ConnectionError:
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

    def test_ocr_processing(self):
        """Test if OCR endpoint correctly processes a document and extracts data."""
        url = f"{self.base_url}/ocr"
        headers = {"Content-Type": "application/json"}
        
        test_data = {
            "file_url": self.test_ocr_url
        }
        
        response = requests.post(url, json=test_data, headers=headers)
        
        # Verify successful response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify OCR response structure
        self.assertIn('extracted_data', data)
        self.assertIn('summary', data)
        
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
        
        # Verify non-empty values
        self.assertTrue(len(extracted_data['Date']) > 0)
        self.assertTrue(len(extracted_data['Amount']) > 0)
        self.assertTrue(len(extracted_data['Vendor/Store']) > 0)
        self.assertTrue(len(extracted_data['Category']) > 0)
        self.assertTrue(len(extracted_data['Description']) > 0)
        self.assertTrue(len(extracted_data['Currency']) > 0)
        self.assertTrue(len(data['summary']) > 0)

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

if __name__ == '__main__':
    unittest.main() 