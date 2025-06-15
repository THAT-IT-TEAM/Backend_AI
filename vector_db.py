import chromadb
from sentence_transformers import SentenceTransformer
import os
import uuid # Import uuid for generating unique IDs
import shutil # Import shutil for directory removal
import io
import pandas as pd
from PyPDF2 import PdfReader
import docx
import json
from typing import Union

# Initialize a local embedding model
# You can choose a different model depending on your needs
# See https://www.sbert.net/docs/pretrained_models.html
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Default persistent DB directory
DEFAULT_DB_DIRECTORY = "./chroma_db"

def process_document_content(content: bytes, file_extension: str) -> str:
    """
    Process document content based on file type and extract text.
    
    Args:
        content: The raw document content in bytes
        file_extension: The file extension (e.g., '.pdf', '.xlsx', etc.)
        
    Returns:
        Extracted text content from the document
    """
    try:
        file_extension = file_extension.lower()
        
        if file_extension == '.txt':
            return content.decode('utf-8')
            
        elif file_extension == '.pdf':
            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
            
        elif file_extension in ['.xls', '.xlsx']:
            excel_file = io.BytesIO(content)
            df = pd.read_excel(excel_file)
            # Convert DataFrame to string representation
            return df.to_string()
            
        elif file_extension in ['.doc', '.docx']:
            doc_file = io.BytesIO(content)
            doc = docx.Document(doc_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
            
        elif file_extension == '.json':
            return json.dumps(json.loads(content), indent=2)
            
        elif file_extension == '.csv':
            csv_file = io.BytesIO(content)
            df = pd.read_csv(csv_file)
            return df.to_string()
            
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        print(f"Error processing document: {e}")
        raise

def get_embedding(text: str):
    """
    Generates an embedding for the given text using the local model.
    """
    return embedding_model.encode(text).tolist()

def get_or_create_vector_db_client(db_directory: str = DEFAULT_DB_DIRECTORY):
    """
    Gets or creates a persistent ChromaDB client.
    """
    return chromadb.PersistentClient(path=db_directory)

def get_or_create_vector_db_collection(db_directory: str = DEFAULT_DB_DIRECTORY):
    """
    Gets or creates a persistent ChromaDB client and returns the collection.
    """
    client = get_or_create_vector_db_client(db_directory)
    collection = client.get_or_create_collection(name="document_chunks")
    return collection, db_directory

def add_document_to_db(document_content: Union[bytes, str], doc_id: str, db_directory: str = DEFAULT_DB_DIRECTORY):
    """
    Adds a document to the specified ChromaDB collection.

    Args:
        document_content: The document content (either as text string or bytes)
        doc_id: A unique ID for the document (should include file extension)
        db_directory: The directory path for the persistent ChromaDB.
    """
    try:
        collection, current_db_directory = get_or_create_vector_db_collection(db_directory)

        # Get file extension from doc_id
        _, file_extension = os.path.splitext(doc_id)
        
        # Process document content based on type
        if isinstance(document_content, bytes):
            text_content = process_document_content(document_content, file_extension)
        else:
            text_content = document_content

        # Split text into chunks with overlap for better context
        chunk_size = 500
        overlap = 100
        chunks = []
        start = 0
        while start < len(text_content):
            end = start + chunk_size
            chunk = text_content[start:end]
            chunks.append(chunk)
            start = end - overlap

        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        embeddings = [get_embedding(chunk) for chunk in chunks]

        # Add chunks and embeddings to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=chunk_ids,
            metadatas=[{"file_type": file_extension[1:], "chunk_index": i} for i in range(len(chunks))]
        )
        print(f"Added {len(chunks)} chunks for document {doc_id} to ChromaDB at {current_db_directory}.")

    except Exception as e:
        print(f"Error adding document {doc_id} to ChromaDB: {e}")
        raise

def search_db(query: str, n_results: int = 5, db_directory: str = DEFAULT_DB_DIRECTORY):
    """
    Searches the specified ChromaDB collection for relevant document chunks.

    Args:
        query: The search query.
        n_results: The number of results to return.
        db_directory: The directory path for the persistent ChromaDB.

    Returns:
        A list of relevant document chunks.
    """
    try:
        # Check if the directory exists before trying to connect
        if not os.path.exists(db_directory):
            print(f"Vector database directory not found: {db_directory}")
            return []

        collection, current_db_directory = get_or_create_vector_db_collection(db_directory)
        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        # Extract the document content from the results
        if results and 'documents' in results and results['documents']:
            return results['documents'][0]
        return []
    except Exception as e:
        print(f"Error searching ChromaDB at {db_directory}: {e}")
        return []

def delete_vector_db(db_directory: str) -> bool:
    """
    Attempts to delete the persistent ChromaDB directory.
    Note: May fail if files are in use by another process (e.g., the running server).
    """
    print(f"Attempting to delete vector database directory: {db_directory}")
    try:
        if os.path.exists(db_directory):
            # Attempt to delete the collection from the client first
            try:
                # Use a temporary client instance for deletion
                client = get_or_create_vector_db_client(db_directory)
                # Check if collection exists before deleting
                if "document_chunks" in [c.name for c in client.list_collections()]:
                    client.delete_collection(name="document_chunks")
                    print(f"Deleted ChromaDB collection 'document_chunks' from client for directory {db_directory}.")
                else:
                    print(f"ChromaDB collection 'document_chunks' not found in client for directory {db_directory}, skipping deletion.")

            except Exception as e:
                print(f"Error deleting ChromaDB collection from client for {db_directory}: {e}")
                # Continue trying to remove the directory even if collection deletion from client fails
                pass

            # Now try to remove the directory
            shutil.rmtree(db_directory)
            print(f"Vector database directory deleted: {db_directory}")
            return True
        else:
            print(f"Vector database directory not found, nothing to delete: {db_directory}")
            return False
    except Exception as e:
        print(f"Error deleting vector database directory {db_directory}: {e}")
        print("Note: This error often occurs if the database files are still in use by the running application.")
        return False

# Example Usage (you can remove or comment this out later)
# if __name__ == "__main__":
#     # Example of adding to the default DB
#     sample_doc_content_default = "This is a sample document for the default DB."
#     doc_identifier_default = "sample_doc_default"
#     add_document_to_db(sample_doc_content_default, doc_identifier_default)
#
#     search_query_default = "what is the default document about?"
#     relevant_chunks_default = search_db(search_query_default)
#     print(f"\nRelevant chunks from default DB for query '{search_query_default}':")
#     for chunk in relevant_chunks_default:
#         print(f"- {chunk}")
#
#     # Example of adding to a different DB (for illustration - requires separate directory)
#     # sample_doc_content_other = "This document goes into a different DB."
#     # doc_identifier_other = "sample_doc_other"
#     # other_db_directory = "./other_chroma_db"
#     # add_document_to_db(sample_doc_content_other, doc_identifier_other, db_directory=other_db_directory)
#
#     # search_query_other = "where does this document go?"
#     # relevant_chunks_other = search_db(search_query_other, db_directory=other_db_directory)
#     # print(f"\nRelevant chunks from other DB for query '{search_query_other}':")
#     # for chunk in relevant_chunks_other:
#     # #     print(f"- {chunk}")
#
#     # Example of deleting a DB
#     # delete_success = delete_vector_db(DEFAULT_DB_DIRECTORY)
#     # if delete_success:
#     #      print(f"Successfully deleted DB at {DEFAULT_DB_DIRECTORY}")
#     # else:
#     #      print(f"Failed to delete DB at {DEFAULT_DB_DIRECTORY}") 