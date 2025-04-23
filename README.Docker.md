# WholeLife Church Chatbot API  Guide

## Server Setup
1. Create a virtual environment and activate it:
```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

2. Install required packages:
```bash
pip install flask PyPDF2 langchain langchain_google_genai google-generativeai faiss-cpu python-dotenv langchain_groq
```

3. Set up Grok api key file:


4. Run the Flask application:
```bash
python3 app.py
```

## API Endpoints Testing in Postman

### 1. Health Check
- **Endpoint**: `GET /health`
- **URL**: `http://localhost:7888/health`
- Expected Response:
```json
{
    "status": "API is running"
}
```

### 2. Get Random Greeting
- **Endpoint**: `GET /start`
- **URL**: `http://localhost:7888/start`
- Expected Response:
```json
{
    "status": "true",
    "start chat": "Random greeting message from the list"
}
```

### 3. Ask Predefined Questions
- **Endpoint**: `POST /df_ask`
- **URL**: `http://localhost:7888/df_ask`
- **Headers**: 
  - Content-Type: application/json
- **Body**:
```json
{
    "question": "I need prayer or pastoral care."
}
```
- Expected Response:
```json
{
    "response": {
        "prayer": "prayer@wholelife.church",
        "pastoral_care": ["freud@wholelife.church", "ken@wholelife.church"]
    }
}
```

### 4. General Questions (RAG-based)
- **Endpoint**: `POST /ask`
- **URL**: `http://localhost:7888/ask`
- **Headers**: 
  - Content-Type: application/json
- **Body**:
```json
{
    "query": "Your question here"
}
```
- Expected Response:
```json
{
    "response": "Answer based on the context from FAISS index"
}
```

## Notes
- Make sure the FAISS index is generated before using the `/ask` endpoint
- The server runs on port 7888 by default
- All responses are in JSON format
- For testing on a remote server, replace `localhost` with the server's IP address