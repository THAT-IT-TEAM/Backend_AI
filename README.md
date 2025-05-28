# That-IT-Team-AI

An AI-powered document processing and chat application that provides OCR capabilities and intelligent document querying.

## Features

- **OCR Processing**: Extract structured data from expense documents using dual OCR methods (Tesseract and LLM)
- **Fraud Detection**: Advanced receipt fraud detection with multiple verification methods
- **Document Chat**: Ask questions about documents using AI
- **Vector Database**: Efficient document storage and retrieval
- **API Endpoints**: RESTful API for all functionalities

## OCR and Fraud Detection

The application uses a dual OCR approach combining traditional Tesseract OCR with LLM-based analysis for enhanced accuracy and fraud detection.

### OCR Methods

1. **Tesseract OCR**

   - Traditional OCR engine for raw text extraction
   - Multiple preprocessing techniques:
     - Basic preprocessing (grayscale, sharpening)
     - Contrast enhancement
     - Denoising
   - Works offline and provides fast processing
   - Best for structured text and clear images

2. **LLM-based OCR (Groq API)**
   - AI-powered text extraction and understanding
   - Direct structured data extraction:
     - Dates, amounts, tax information
     - Vendor details and items
     - Payment methods and document IDs
   - Better context understanding
   - Handles poor quality images effectively
   - Requires internet connection

### Fraud Detection Features

1. **Image Analysis**

   - Image quality assessment
   - Manipulation detection:
     - Copy-move forgery
     - JPEG compression analysis
     - Noise pattern analysis
     - Edge consistency
     - Metadata verification

2. **Text Analysis**

   - Pattern recognition for suspicious elements
   - Format consistency checks
   - Multiple decimal points detection
   - Inconsistent spacing analysis
   - Suspicious character detection

3. **Verification Methods**

   - Vendor information verification
   - Amount validation
   - Date and time verification
   - Location verification
   - Category-specific checks

4. **Risk Assessment**
   - Overall risk score calculation
   - Fraud probability estimation
   - Confidence scoring
   - Detailed risk factor analysis

## API Endpoints

### 1. OCR Processing (`/ocr`)

Processes documents and extracts structured expense data using dual OCR methods.

**Endpoint:** `POST /ocr`

**Request Body:**

```json
{
  "file_url": "https://example.com/document.jpg",
  "user_id": "uuid-string",
  "trip_id": "uuid-string"
}
```

**Response:**

```json
{
  "expense_id": "uuid-string",
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
  "ocr_confidence": 0.95,
  "summary": "Summary of the expense..."
}
```

### 2. Fraud Detection (`/fraud-check`)

Analyzes receipts for potential fraud using multiple verification methods.

**Endpoint:** `POST /fraud-check`

**Request Body:**

```json
{
  "expense_id": "uuid-string",
  "file_url": "https://example.com/document.jpg"
}
```

**Response:**

```json
{
  "fraud_check_id": "uuid-string",
  "overall_risk_score": 0.15,
  "fraud_probability": 0.12,
  "risk_factors": [
    "Low risk: All verifications passed",
    "Image quality: Good",
    "Text consistency: Verified"
  ],
  "verification_results": {
    "vendor_verification": { "status": "verified" },
    "amount_verification": { "status": "verified" },
    "date_verification": { "status": "verified" }
  },
  "image_analysis_results": {
    "quality_score": 0.95,
    "manipulation_indicators": []
  },
  "summary": "Low risk receipt with verified details"
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

### 3. Analytics Endpoints

#### Get Trip Analytics (`/api/analytics/trip`)

Get comprehensive analytics for a specific trip.

**Endpoint:** `POST /api/analytics/trip`

**Request Body:**

```json
{
  "trip_name": "Summer Vacation 2024"
}
```

**Response:**

```json
{
  "expense_distribution": "Plotly figure JSON",
  "trend_analysis": "Plotly figure JSON",
  "budget_comparison": "Plotly figure JSON",
  "expense_clusters": "Plotly figure JSON",
  "ai_insights": "AI-generated insights about the trip expenses"
}
```

#### Get All Analytics (`/api/analytics/all`)

Get analytics for all trips in the system.

**Endpoint:** `GET /api/analytics/all`

**Response:**

```json
{
  "expense_distribution": "Plotly figure JSON",
  "trend_analysis": "Plotly figure JSON",
  "expense_clusters": "Plotly figure JSON"
}
```

### Server Cleanup

The server implements graceful shutdown and cleanup procedures:

1. **Vector Database Cleanup**

   - Vector databases marked for deletion are cleaned up on server shutdown
   - Use the `/delete_db` endpoint to mark databases for deletion
   - Cleanup is handled automatically on server exit

2. **Ngrok Tunnel Cleanup**

   - If ngrok is configured, the tunnel is properly disconnected on shutdown
   - Handles both normal shutdown and interrupt signals (SIGINT, SIGTERM)
   - Ensures no lingering tunnels remain active

3. **Signal Handling**
   - Implements proper signal handlers for graceful shutdown
   - Handles SIGINT (Ctrl+C) and SIGTERM signals
   - Ensures all cleanup procedures are executed before exit

Example usage of analytics endpoints:

```bash
# Get analytics for a specific trip
curl -X POST http://localhost:8080/api/analytics/trip \
  -H "Content-Type: application/json" \
  -d '{"trip_name": "Summer Vacation 2024"}'

# Get analytics for all trips
curl -X GET http://localhost:8080/api/analytics/all \
  -H "Accept: application/json"
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
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
NGROK_AUTH_TOKEN=your_ngrok_auth_token
TESSERACT_CMD=path_to_tesseract_executable  # Optional
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
- `pytesseract`: For OCR processing
- `Pillow`: For image processing
- `opencv-python`: For image analysis
- `groq`: For LLM-based OCR
- `supabase`: For database operations

## Usage Examples

### OCR Processing

```bash
curl -X POST https://your-api-url.ngrok-free.app/ocr \
-H "Content-Type: application/json" \
-d '{
    "file_url": "https://example.com/document.jpg",
    "user_id": "uuid-string",
    "trip_id": "uuid-string"
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
