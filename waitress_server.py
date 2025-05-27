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

# import necessary modules
from supabase_fetch import fetch_document_from_supabase, supabase # Import supabase client
from vector_db import add_document_to_db, search_db, delete_vector_db, DEFAULT_DB_DIRECTORY # Import delete_vector_db and DEFAULT_DB_DIRECTORY
from llm_interaction import get_chatbot_response

# For document processing (placeholders - install necessary libraries)
# from PyPDF2 import PdfReader
# import pandas as pd
# from PIL import Image
# import pytesseract # For OCR

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# List to hold vector database directories to be deleted on exit
dbs_to_delete_on_exit = []

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

        # 1. Fetch document from Supabase
        document_content = fetch_document_from_supabase(bucket_name, document_id)
        if document_content is None:
            return jsonify({"error": f"Could not fetch document '{document_id}' from Supabase bucket '{bucket_name}'."}), 500

        # 2. Process document and add to the NEW vector DB (logs to Supabase)
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

    supabase_delete_successful = False
    # 1. Delete records from Supabase table
    try:
        # Supabase delete returns data and count. Check data for success.
        data, count = supabase.table('vector_db_documents').delete().eq('vector_db_name', vector_db_name).execute()
        if data:
             print(f"Deleted records from Supabase table 'vector_db_documents' for DB {vector_db_name}.")
             supabase_delete_successful = True
        else:
             print(f"No matching records found or deleted in Supabase table 'vector_db_documents' for DB {vector_db_name}.")

    except Exception as e:
        print(f"Error deleting Supabase records for DB {vector_db_name}: {e}")
        # Continue, as the main action for this endpoint is marking for deletion
        pass

    if supabase_delete_successful:
        return jsonify({"message": f"Supabase records for '{vector_db_name}' deleted. Vector database directory deletion scheduled for shutdown."}), 200
    else:
        # Return 200 even if Supabase deletion failed, as the primary action (marking for deletion) succeeded
        return jsonify({"message": f"No matching Supabase records found or deleted for '{vector_db_name}'. Vector database directory deletion scheduled for shutdown."}), 200

# --- Server Execution ---

# Cleanup function to delete marked databases on exit
def cleanup_dbs_on_exit():
    print("Checking for databases to delete on exit...")
    for db_directory in dbs_to_delete_on_exit:
        delete_vector_db(db_directory)
    print("Database cleanup complete.")

# Cleanup function to disconnect ngrok
def cleanup_ngrok():
    print("Disconnecting ngrok...")
    try:
        ngrok.disconnect()
        print("Ngrok disconnected.")
    except Exception as e:
        print(f"Error during ngrok disconnection: {e}")

# Register cleanup functions (registered in reverse order of execution)
atexit.register(cleanup_ngrok)
atexit.register(cleanup_dbs_on_exit)

if __name__ == '__main__':
    # Get Ngrok auth token and domain from environment variables
    ngrok_auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    # You'll need to add an NGROK_DOMAIN variable to your .env file for a fixed domain
    ngrok_domain = os.environ.get("NGROK_DOMAIN")

    print("Starting Waitress server on http://127.0.0.1:5000")

    # Start Waitress server in a separate thread so Ngrok can run in the main thread
    import threading
    server_thread = threading.Thread(target=lambda: serve(app, host='127.0.0.1', port=5000))
    server_thread.daemon = True # Allow main thread to exit even if server thread is running
    server_thread.start()

    # Give the server a moment to start
    time.sleep(2) # Increased sleep time slightly

    # Check if the server thread is alive. If not, there was likely an error starting Waitress.
    if not server_thread.is_alive():
        print("Error: Waitress server failed to start. Exiting.")
        exit(1)


    if not ngrok_auth_token:
        print("Warning: NGROK_AUTH_TOKEN not set in .env. Ngrok tunnel will not be created automatically.")
        # If ngrok is not configured, keep the local server running
        server_thread.join() # Wait for the server thread to finish
    else:
        try:
            # Set Ngrok auth token
            ngrok.set_auth_token(ngrok_auth_token)
            print("Ngrok auth token set.")

            # Connect to the specified port (5000) using ngrok.forward
            print("Starting ngrok tunnel...")

            # Check if ngrok_domain is set and not empty
            if ngrok_domain:
                 listener = ngrok.forward(5000, domain=ngrok_domain)
            else:
                 listener = ngrok.forward(5000)

            # Check if listener is valid before accessing url()
            if listener:
                 public_url = listener.url()
                 print(f"Ngrok tunnel established at: {public_url}")
                 print("Waitress server and Ngrok tunnel are running.")
                 print("Press Ctrl+C to stop.")

                 # Keep the main thread alive to keep the ngrok tunnel active
                 while True:
                    time.sleep(0.1) # Reduced sleep to be more responsive to Ctrl+C
            else:
                 # If listener is not created, this part will be reached.
                 print("Ngrok listener not created.")
                 print("The server is running locally, but not publicly accessible via Ngrok.")
                 # Keep the main thread alive for the server_thread to run.
                 server_thread.join() # Wait for the server thread to finish

        except Exception as e:
            print(f"Error starting ngrok tunnel: {e}")
            print("The server is running locally, but not publicly accessible via Ngrok.")
            # If ngrok fails in the try block, this part will be reached.
            # Keep the main thread alive for the server_thread to run.
            server_thread.join() # Wait for the server thread to finish if ngrok failed

    # This part will be reached when the server_thread finishes (either after ngrok failure or if ngrok wasn't configured).
    # If ngrok succeeded, the while True loop prevented reaching here until script termination (e.g. Ctrl+C),
    # at which point atexit handlers would run.
    pass # Keep the script running if the server thread is active after ngrok issues 