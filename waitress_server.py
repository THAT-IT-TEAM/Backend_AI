import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from waitress import serve
import ngrok
import atexit # Import atexit for cleanup
import time # Import time for potential delays
import io # Import io for handling bytes data
import uuid # Import uuid for generating unique IDs
import re # Import re for sanitizing directory names
import asyncio
from uuid import UUID
from typing import Dict, Any
from flask.views import View
from flask.typing import ResponseReturnValue
from asgiref.wsgi import WsgiToAsgi
from hypercorn.asyncio import serve as hypercorn_serve
from hypercorn.config import Config
import signal
import sys
import plotly.express as px
import plotly.graph_objects as go

# import necessary modules
from api_client import fetch_document_from_api, delete_vector_db_document_via_api # Import from our new API client
from vector_db import add_document_to_db, search_db, delete_vector_db, DEFAULT_DB_DIRECTORY # Import delete_vector_db and DEFAULT_DB_DIRECTORY
from llm_interaction import get_chatbot_response
from ocr_expense_parser import parse_expense_text
from receipt_fraud_detector import ReceiptFraudDetector, check_receipt_fraud
from trip_analytics import TripAnalytics

# For document processing (placeholders - install necessary libraries)
# from PyPDF2 import PdfReader
# import pandas as pd
# from PIL import Image
# import pytesseract # For OCR

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True

# Configure async support
asgi_app = WsgiToAsgi(app)

# List to hold vector database directories to be deleted on exit
dbs_to_delete_on_exit = []

# Global variables
ngrok_tunnel = None

# --- Document Processing and Indexing ----
#
# This is a simplified approach. In a real application, you would want a
# more robust way to handle different document types (PDFs, images, excels)
# and manage the vector database (e.g., check if a document has already
# been indexed, handle updates, use a persistent ChromaDB).
#
# You will need libraries like:
# - PyPDF2 or pdfminer.six for PDFs
# - Pillow for images (you might need OCR)
# - pandas for Excel files
#
# And logic to extract text content from these file types.

def extract_text_from_document(document_content: bytes, file_extension: str) -> str:
    """
    Extracts text content from document bytes based on file extension.
    Implement logic for different file types here.
    """
    text_content = ""
    try:
        if file_extension == '.txt':
            text_content = document_content.decode('utf-8')
        elif file_extension == '.pdf':
            # Example using PyPDF2 (install with: pip install pypdf2)
            # reader = PdfReader(io.BytesIO(document_content))
            # text_content = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
            print("PDF processing not implemented. Install PyPDF2 and uncomment code.")
            pass # Placeholder for PDF processing
        elif file_extension in ['.xls', '.xlsx']:
             # Example using pandas (install with: pip install pandas openpyxl)
             # df = pd.read_excel(io.BytesIO(document_content))
             # text_content = df.to_string()
             print("Excel processing not implemented. Install pandas/openpyxl and uncomment code.")
             pass # Placeholder for Excel processing
        elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
             # Example using Pillow and pytesseract (install with: pip install Pillow pytesseract)
             # Image processing usually requires OCR
             # img = Image.open(io.BytesIO(document_content))
             # text_content = pytesseract.image_to_string(img)
             print("Image processing not implemented (requires OCR). Install Pillow/pytesseract and uncomment code.")
             pass # Placeholder for Image processing
        else:
            print(f"Unsupported file extension: {file_extension}")

    except Exception as e:
        print(f"Error extracting text from document: {e}")
        text_content = ""

    return text_content

def process_document_content(document_content: bytes, doc_id: str, db_directory: str) -> bool:
    """
    Processes raw document content (bytes), extracts text, and adds to the specified vector DB.
    Determines file type and extracts text before adding to the specified DB.
    """
    if document_content is None:
        print("No document content to process.")
        return False

    # Determine file extension from doc_id (assuming doc_id includes extension)
    _, file_extension = os.path.splitext(doc_id)
    file_extension = file_extension.lower()

    text_content = extract_text_from_document(document_content, file_extension)

    if not text_content:
        print(f"Could not extract text from document {doc_id}. Text extraction failed or document is empty.")
        return False

    try:
        # Pass the db_directory to add_document_to_db
        add_document_to_db(text_content, doc_id, db_directory=db_directory)
        return True
    except Exception as e:
        print(f"Error processing document {doc_id} for DB {db_directory}: {e}")
        return False

# --- API Endpoints ---

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles incoming chat requests.
    Expects JSON with either:
    1. 'vector_db_name' and 'question' (Queries an existing DB)
    2. 'document_id', 'bucket_name', and 'question' (Fetches, processes into a NEW DB, and queries)
    """
    data = request.get_json()

    vector_db_name = data.get('vector_db_name')
    document_id = data.get('document_id')
    bucket_name = data.get('bucket_name')
    question = data.get('question')

    if not question:
        return jsonify({"error": "'question' is required."}), 400

    relevant_chunks = []
    current_db_directory_used = None # To track which DB was used/created

    if vector_db_name:
        print(f"Received request for existing DB '{vector_db_name}' with question: '{question}'")
        # Case 1: vector_db_name is provided, search this DB
        relevant_chunks = search_db(question, n_results=5, db_directory=vector_db_name)
        current_db_directory_used = vector_db_name

    elif document_id and bucket_name:
        print(f"Received request for new document '{document_id}' in bucket '{bucket_name}' with question: '{question}'")
        # Case 2: document_id and bucket_name are provided, fetch, process into a NEW DB, and search

        # Generate a unique directory name for this document's vector DB
        # Sanitize doc_id to be filesystem-safe and append a unique ID
        sanitized_doc_id = re.sub(r'[^a-zA-Z0-9_.-]', '_', document_id)
        new_db_directory = f"./chroma_db_{sanitized_doc_id}_{uuid.uuid4().hex}"
        current_db_directory_used = new_db_directory

        print(f"Processing document into new vector database: {new_db_directory}")

        # 1. Fetch document from API
        document_content = fetch_document_from_api(bucket_name, document_id)
        if document_content is None:
            return jsonify({"error": f"Could not fetch document '{document_id}' from API bucket '{bucket_name}'."}), 500

        # 2. Process document and add to the NEW vector DB
        processed = process_document_content(document_content, document_id, db_directory=new_db_directory)
        if not processed:
             # If processing fails, the new directory might still exist but be incomplete or empty.
             # We should probably mark it for deletion or handle cleanup.
             # For now, we'll just return an error.
             # Optionally add new_db_directory to dbs_to_delete_on_exit here if processing fails.
             return jsonify({"error": f"Could not process document '{document_id}'. Text extraction failed or document is empty."}), 500

        # 3. Search the NEW vector DB for relevant context
        relevant_chunks = search_db(question, n_results=5, db_directory=new_db_directory)

    else:
        # Neither case is met
        return jsonify({"error": "Invalid request body. Provide either 'vector_db_name' or both 'document_id' and 'bucket_name'."}), 400

    context = "\n".join(relevant_chunks)

    if not context:
        print(f"No relevant context found for the given input and question '{question}' in DB {current_db_directory_used}.")
        # Optionally, you could still send the question to the LLM without context
        # or return a specific message.
        # For now, we'll inform the user if no context was found.
        return jsonify({"response": f"Could not find relevant information in the specified document or database ('{current_db_directory_used}') to answer your question."})

    # 4. Get chatbot response from Groq
    chatbot_response = get_chatbot_response(question, context)

    # Optionally return the DB name used if a new one was created
    response_data = {"response": chatbot_response}
    if document_id and bucket_name and current_db_directory_used:
        response_data["vector_db_name_used"] = current_db_directory_used

    return jsonify(response_data)

@app.route('/delete_db', methods=['POST'])
def delete_db_endpoint():
    """
    Handles requests to mark a vector database for deletion and deletes its Supabase records.
    The actual directory deletion happens on script exit.
    Expects JSON with 'vector_db_name'.
    """
    data = request.get_json()

    vector_db_name = data.get('vector_db_name')

    if not vector_db_name:
        return jsonify({"error": "'vector_db_name' is required in the request body."}), 400

    print(f"Received request to mark vector database for deletion: {vector_db_name}")

    # Mark the database for deletion on exit
    if vector_db_name not in dbs_to_delete_on_exit:
        dbs_to_delete_on_exit.append(vector_db_name)
        print(f"Vector database '{vector_db_name}' marked for deletion on exit.")
    else:
        print(f"Vector database '{vector_db_name}' was already marked for deletion.")

    # 1. Delete records from Node.js API
    api_delete_successful = delete_vector_db_document_via_api(vector_db_name)
    if not api_delete_successful:
        print(f"Warning: Failed to delete records from Node.js API for DB {vector_db_name}.")
        # Even if API deletion fails, we still proceed with marking for local deletion

    return jsonify({"message": f"Vector database '{vector_db_name}' marked for local deletion and API record deletion attempted.", "api_delete_successful": api_delete_successful})

@app.route('/ocr', methods=['POST'])
async def process_ocr():
    """
    Handles OCR processing of documents from URLs.
    Expects JSON with:
    - file_url: URL of the document to process
    - user_id: UUID of the user submitting the expense
    - trip_id: UUID of the trip this expense belongs to
    """
    data = request.get_json()

    file_url = data.get('file_url')
    user_id_str = data.get('user_id')
    trip_id_str = data.get('trip_id')

    if not file_url or not user_id_str or not trip_id_str:
        return jsonify({"error": "Missing file_url, user_id, or trip_id"}), 400

    try:
        user_id = UUID(user_id_str)
        trip_id = UUID(trip_id_str)
    except ValueError:
        return jsonify({"error": "Invalid user_id or trip_id format (must be UUID)"}), 400

    print(f"Received OCR request for file: {file_url}, user: {user_id}, trip: {trip_id}")
    
    try:
        # The key change: await parse_expense_text
        parsed_result = await parse_expense_text(file_url, user_id, trip_id)
        if parsed_result['expense_id']:
            return jsonify({"message": "OCR processed and expense stored successfully", "expense_id": parsed_result['expense_id'], "summary": parsed_result['summary']}), 200
        else:
            # If expense_id is None, it means storage failed or parsing was incomplete
            return jsonify({"error": parsed_result.get('summary', "Failed to process OCR and store expense data.")}), 500

    except Exception as e:
        print(f"Error processing OCR request: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({"error": f"Error processing OCR: {e}"}), 500

@app.route('/fraud-check', methods=['POST'])
async def check_fraud():
    """
    Handles fraud detection for expense receipts.
    Expects JSON with:
    - expense_id: ID of the expense to check
    - file_url: URL of the receipt image/document
    
    Returns fraud analysis results in JSON format.
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    required_fields = ['expense_id', 'file_url']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
    
    expense_id = data['expense_id']
    file_url = data['file_url']
    
    if not expense_id:
        return jsonify({"error": "Invalid or missing expense_id"}), 400
        
    if not file_url:
        return jsonify({"error": "Empty file URL provided"}), 400
        
    try:
        # Validate expense_id is a valid UUID
        try:
            expense_id = UUID(expense_id)
        except ValueError:
            return jsonify({"error": "Invalid UUID format for expense_id"}), 400

        # Create fraud detector instance and analyze receipt
        result = await check_receipt_fraud(expense_id, file_url)
        
        if not result:
            return jsonify({"error": "Failed to analyze receipt for fraud"}), 500
            
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing fraud check request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/trip', methods=['POST'])
def get_trip_analytics():
    """Get all analytics for a specific trip by name from request body"""
    try:
        data = request.get_json()
        if not data or 'trip_name' not in data:
            return jsonify({'error': 'Missing trip_name in request body'}), 400
            
        trip_name = data['trip_name']
        print(f"Processing analytics for trip: {trip_name}")  # Debug log
        
        analytics = TripAnalytics()
        results = analytics.get_all_analytics(trip_name)
        print(f"Analytics results keys: {list(results.keys())}")  # Debug log
        
        # Convert Plotly figures to JSON
        for key, value in results.items():
            print(f"Processing key: {key}, value type: {type(value)}")  # Debug log
            if hasattr(value, 'to_json'):  # Check if it's a Plotly figure by checking for to_json method
                results[key] = value.to_json()
            elif value is not None:  # Debug log for non-None values that aren't figures
                print(f"Non-figure value for {key}: {value}")
        
        return jsonify(results)
    except Exception as e:
        print(f"Error in get_trip_analytics: {str(e)}")  # Debug log
        import traceback
        print(f"Traceback: {traceback.format_exc()}")  # Debug log with full traceback
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/all', methods=['GET'])
def get_all_analytics():
    """Get analytics for all trips"""
    try:
        analytics = TripAnalytics()
        results = analytics.get_all_analytics()
        
        # Convert Plotly figures to JSON
        for key, value in results.items():
            if hasattr(value, 'to_json'):  # Check if it's a Plotly figure by checking for to_json method
                results[key] = value.to_json()
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Server Execution ---

# Cleanup function to delete marked databases on exit
def cleanup_dbs_on_exit():
    print("Checking for databases to delete on exit...")
    for db_directory in dbs_to_delete_on_exit:
        delete_vector_db(db_directory)
    print("Database cleanup complete.")

# Cleanup function to disconnect ngrok
def cleanup_ngrok():
    """Clean up ngrok tunnel if it exists."""
    global ngrok_tunnel
    try:
        if ngrok_tunnel is not None and hasattr(ngrok_tunnel, 'public_url'):
            print(f"Disconnecting ngrok tunnel at {ngrok_tunnel.public_url}")
            ngrok.disconnect(ngrok_tunnel.public_url)
        else:
            print("No active ngrok tunnel to disconnect")
    except Exception as e:
        print(f"Error during ngrok disconnection: {str(e)}")
    finally:
        ngrok_tunnel = None
        print("Ngrok disconnected.")

# Register cleanup functions (registered in reverse order of execution)
atexit.register(cleanup_ngrok)
atexit.register(cleanup_dbs_on_exit)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\nReceived shutdown signal. Cleaning up...")
    cleanup_ngrok()
    print("Shutdown complete.")
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    async def run_server_async():
        # Create a new event loop for Hypercorn to run in if not already present
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Configure Hypercorn
        hypercorn_config = Config()
        hypercorn_config.bind = [f"0.0.0.0:{os.environ.get('FLASK_PORT', '8080')}"]

        try:
            # Set up ngrok tunnel if configured
            ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN")
            ngrok_domain = os.environ.get("NGROK_DOMAIN")

            if ngrok_auth_token:
                from pyngrok import ngrok, conf
                conf.get_default().auth_token = ngrok_auth_token

                if ngrok_domain:
                    ngrok_tunnel = ngrok.connect(8080, domain=ngrok_domain)
                else:
                    ngrok_tunnel = ngrok.connect(8080)

                print(f"Ngrok tunnel established at: {ngrok_tunnel.public_url}")
            else:
                print("Warning: NGROK_AUTH_TOKEN not set. Running without ngrok tunnel.")

            # Start the server using Hypercorn
            await hypercorn_serve(asgi_app, hypercorn_config)

        except Exception as e:
            print(f"Error starting server: {str(e)}")
            cleanup_ngrok()
            sys.exit(1)
        finally:
            # Clean up databases on exit
            print("Checking for databases to delete on exit...")
            cleanup_dbs_on_exit()
            print("Database cleanup complete.")
            # Clean up ngrok
            print("Disconnecting ngrok...")
            cleanup_ngrok()

    # Run the async server function
    asyncio.run(run_server_async()) 