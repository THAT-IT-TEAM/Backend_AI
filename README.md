# AI Chatbot with Supabase, ChromaDB, and Groq

This project implements a simple AI chatbot that can fetch documents from Supabase storage, create a vector database using ChromaDB with a local embedding model, and answer questions about the documents using the Groq API. The application is served using Waitress and exposed via Ngrok.

## Project Structure

```
.
├── .env
├── supabase_fetch.py
├── vector_db.py
├── llm_interaction.py
├── waitress_server.py
├── requirements.txt
└── README.md
```

## Setup

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Python virtual environment (recommended):**

    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Supabase:**

    - Create a new project in Supabase.
    - Go to the Storage section and create a new bucket (e.g., `data-storage`).
    - Go to the Database section and create a new table named `vector_db_documents` with the following schema:

      ```sql
      CREATE TABLE public.vector_db_documents (
          id UUID PRIMARY KEY UNIQUE,
          vector_db_name TEXT,
          document_id TEXT,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
      );
      ```

5.  **Get API Keys:**

    - **Supabase:** Find your Project URL and `anon` key in your Supabase project settings under `API`. The key is under `project api keys` -> `anon` -> `eyJ...`.
    - **Groq:** Sign up for Groq Cloud and generate an API key.
    - **Ngrok (Optional for public access):** Sign up for Ngrok and get your auth token from the dashboard. If you have a paid plan and a reserved domain, note that as well.

6.  **Create and configure the `.env` file:**

    Create a file named `.env` in the root directory of your project and add the following content, replacing the placeholder values with your actual keys:

    ```dotenv
    # Supabase credentials
    SUPABASE_URL=your_supabase_url
    SUPABASE_KEY=your_supabase_key

    # Groq API key
    GROQ_API_KEY=your_groq_api_key

    # Optional: Ngrok auth token for exposing the server publicly
    NGROK_AUTH_TOKEN=your_ngrok_auth_token

    # Optional: Ngrok domain for a fixed public URL (requires paid Ngrok plan)
    # NGROK_DOMAIN=your_fixed_ngrok_domain
    ```

## Running the Server

1.  Make sure your Python virtual environment is activated.
2.  Run the `waitress_server.py` script:

    ```bash
    python waitress_server.py
    ```

    The server will start using Waitress on `http://127.0.0.1:5000`. If `NGROK_AUTH_TOKEN` is provided in your `.env`, it will also attempt to create an Ngrok tunnel and print the public URL.

## API Endpoints

The server exposes two main API endpoints:

### `POST /chat`

This endpoint handles chat requests. It can either process a new document and answer a question, or answer a question based on an already indexed vector database.

**Request Body (JSON):**

Requires a `question` key and one of the following combinations:

1.  **To process a new document and ask a question:**

    ```json
    {
      "document_id": "path/to/your/document_in_supabase.ext",
      "bucket_name": "your_supabase_bucket",
      "question": "Your question goes here?"
    }
    ```

    - `document_id`: The path to the document within your Supabase storage bucket.
    - `bucket_name`: The name of your Supabase storage bucket.
    - `question`: The question to ask about the document.

    _Note: This will create a new vector database directory locally for this document (named based on the `document_id` and a unique ID) and log it in the Supabase `vector_db_documents` table. The response will include the `vector_db_name_used`._

2.  **To answer a question based on an already indexed vector database:**

    ```json
    {
      "vector_db_name": "the_name_of_the_vector_db_to_query",
      "question": "Your question goes here?"
    }
    ```

    - `vector_db_name`: The name (directory path) of the vector database to query (e.g., `./chroma_db_report_txt_<unique_id>`). You can get this from the response when processing a new document or from your Supabase `vector_db_documents` table.
    - `question`: The question to ask about the documents within this vector database.

### `POST /delete_db`

This endpoint marks a vector database directory for deletion on script exit and deletes its associated records from the Supabase `vector_db_documents` table.

**Request Body (JSON):**

Requires a `vector_db_name` key:

```json
{
  "vector_db_name": "the_name_of_the_vector_db_to_delete"
}
```

- `vector_db_name`: The name (directory path) of the vector database to delete (e.g., `./chroma_db_report_txt_<unique_id>`). This directory will be deleted when the `waitress_server.py` script is stopped.

## Implementation Details

- **`.env`:** Stores sensitive information like API keys and Supabase credentials.
- **`supabase_fetch.py`:** Handles connecting to Supabase and fetching document content from a specified bucket and file path.
- **`vector_db.py`:** Manages the ChromaDB vector database. It uses a persistent client storing data in local directories. It handles getting/creating databases, generating embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`), adding document chunks, and performing similarity searches. It also logs document additions to the Supabase `vector_db_documents` table and includes a function to attempt deletion of a vector database directory.
- **`llm_interaction.py`:** Handles communication with the Groq API. It takes a question and relevant context (document chunks) and uses the Groq API to generate a natural language response.
- **`waitress_server.py`:** This is the main application file. It sets up a Flask application served by Waitress. It defines the `/chat` and `/delete_db` API endpoints. It orchestrates the workflow by calling functions from the other modules based on the incoming requests. It also uses `pyngrok` to optionally expose the server publicly and registers cleanup functions with `atexit` to disconnect Ngrok and attempt to delete marked vector database directories upon script exit.
- **`requirements.txt`:** Lists the Python dependencies required for the project.

## Document Type Handling

The `waitress_server.py` includes a placeholder function `extract_text_from_document` for handling different document types (PDFs, images, Excel). **Currently, only basic `.txt` file processing is fully implemented.** To support other formats, you will need to install additional libraries (e.g., `PyPDF2`, `Pillow`, `pytesseract`, `pandas`, `openpyxl`) and add the necessary text extraction logic within this function.

## Known Issues

- **Vector Database Directory Deletion on Windows:** Due to file locking behavior on Windows, the automatic deletion of vector database directories via the `atexit` cleanup function may sometimes fail with a "file in use" error (`WinError 32`). If this occurs, you may need to manually delete the directory after stopping the `waitress_server.py` script.
