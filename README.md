# That-IT-Team-AI

An AI-powered document processing and chat application that provides OCR capabilities and intelligent document querying.

## Features

- **OCR Processing**: Extract structured data from expense documents
- **Document Chat**: Ask questions about documents using AI
- **Vector Database**: Efficient document storage and retrieval
- **API Endpoints**: RESTful API for all functionalities

## API Endpoints

### 1. OCR Processing (`/ocr`)

Processes documents and extracts structured expense data.

**Endpoint:** `POST /ocr`

**Request Body:**

```json
{
  "file_url": "https://example.com/document.jpg"
}
```

**Response:**

```json
{
  "extracted_data": {
    "Date": "2024-03-20",
    "Amount": "₹618.50",
    "Vendor/Store": "Uber",
    "Category": "Transportation",
    "Description": "Trip fare details...",
    "Currency": "INR",
    "Tax Amount": "₹29.46",
    "Document ID or Reference Number": null,
    "Payment Method": null
  },
  "summary": "Summary of the expense..."
}
```

### 2. Document Chat (`/chat`)

Query documents using natural language.

**Endpoint:** `POST /chat`

**Request Body (New Document):**

```json
{
  "document_id": "report.txt",
  "bucket_name": "data-storage",
  "question": "What is the main topic of this document?"
}
```

**Request Body (Existing Database):**

```json
{
  "vector_db_name": "path/to/vector/db",
  "question": "What are the key points?"
}
```

**Response:**

```json
{
  "response": "AI-generated answer...",
  "vector_db_name_used": "path/to/vector/db"
}
```

## Testing

The application includes comprehensive test cases to verify core functionalities:

### Running Tests

```bash
python -m unittest tests/test_api.py
```

### Test Cases

1. **OCR Processing Test**

   - Verifies document processing
   - Validates extracted data structure
   - Checks data types and non-empty values
   - Required fields: Date, Amount, Vendor/Store, Category, Description, Currency, Tax Amount

2. **Document Chat Test**
   - Tests document processing and vector DB creation
   - Verifies question answering capability
   - Tests follow-up questions
   - Validates response structure and content

## Recent Changes

1. **OCR Processing**

   - Updated to handle document URLs directly
   - Enhanced data extraction with additional fields
   - Improved error handling
   - Added structured response format

2. **Document Processing**

   - Simplified vector database creation
   - Enhanced question answering capabilities
   - Added support for follow-up questions
   - Improved response formatting

3. **Testing**
   - Streamlined test suite to focus on core functionalities
   - Added comprehensive OCR response validation
   - Enhanced chat endpoint testing
   - Improved error handling tests

## Environment Setup

1. Create a `.env` file with the following variables:

```env
API_URL=https://your-api-url.ngrok-free.app
GROQ_API_KEY=your_groq_api_key
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

Key dependencies include:

- `requests`: For API calls
- `python-dotenv`: For environment variable management
- `chromadb`: For vector database
- `sentence-transformers`: For text embeddings
- `pandas`: For data processing
- `PyPDF2`: For PDF handling

## Usage Examples

### OCR Processing

```bash
curl -X POST https://your-api-url.ngrok-free.app/ocr \
-H "Content-Type: application/json" \
-d '{
    "file_url": "https://example.com/document.jpg"
}'
```

### Document Chat

```bash
curl -X POST https://your-api-url.ngrok-free.app/chat \
-H "Content-Type: application/json" \
-d '{
    "document_id": "report.txt",
    "bucket_name": "data-storage",
    "question": "What is the main topic?"
}'
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
