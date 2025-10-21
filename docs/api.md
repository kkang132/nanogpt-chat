# API Documentation

## Base URL

```
http://localhost:5000
```

## Endpoints

### GET /

**Description**: Serves the chat web interface.

**Response**:
- **Content-Type**: `text/html`
- **Body**: HTML page (`templates/index.html`)

**Example**:
```bash
curl http://localhost:5000/
```

---

### POST /chat

**Description**: Send a message and receive a model-generated response.

#### Request

**Content-Type**: `application/json`

**Body**:
```json
{
  "message": "string"  // User's message (required)
}
```

**Example**:
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'
```

#### Response

**Success (200 OK)**:
```json
{
  "response": "string",    // Model's generated response
  "chat_count": number     // Total conversations saved
}
```

**Example Response**:
```json
{
  "response": "Machine learning is a branch of AI that enables computers to learn from data.",
  "chat_count": 42
}
```

**Error (400 Bad Request)**:
```json
{
  "error": "No message provided"
}
```

#### Behavior

1. Receives user message
2. Formats as `"Human: {message}\nAssistant:"`
3. Generates response using GPT-2 model
4. Saves conversation to `chat_history.jsonl`
5. Returns response with updated chat count

#### Generation Parameters

The endpoint uses the following generation settings (hardcoded in `app.py`):
- `max_tokens`: 150
- `temperature`: 0.8
- `top_k`: 200

To modify these, edit line 122 in `app.py`:
```python
response = generate_response(prompt, max_tokens=150, temperature=0.8)
```

---

### GET /stats

**Description**: Get statistics about collected chat data.

#### Request

No parameters required.

**Example**:
```bash
curl http://localhost:5000/stats
```

#### Response

**Success (200 OK)**:
```json
{
  "chat_count": number,              // Total conversations collected
  "ready_for_finetuning": boolean    // True if chat_count >= 10
}
```

**Example Response** (not ready):
```json
{
  "chat_count": 5,
  "ready_for_finetuning": false
}
```

**Example Response** (ready):
```json
{
  "chat_count": 25,
  "ready_for_finetuning": true
}
```

#### Behavior

- Counts lines in `chat_history.jsonl`
- Returns `chat_count: 0` if file doesn't exist
- Sets `ready_for_finetuning: true` when `chat_count >= 10`

---

## CORS Configuration

**CORS is enabled** for all origins using `flask-cors`:

```python
from flask_cors import CORS
CORS(app)
```

This allows the API to be called from any web application.

---

## Error Handling

### Client Errors (4xx)

**400 Bad Request**: Missing or invalid request data
```json
{
  "error": "No message provided"
}
```

### Server Errors (5xx)

**500 Internal Server Error**: Unhandled exceptions
- Model loading failures
- Generation errors
- File I/O errors

Errors are logged to console but may not return structured JSON responses.

---

## Data Persistence

### Chat History

All conversations are appended to `chat_history.jsonl` in JSON Lines format:

```json
{"timestamp": "2025-10-21T16:30:00.123456", "user": "Hello", "assistant": "Hi there!"}
{"timestamp": "2025-10-21T16:30:15.789012", "user": "How are you?", "assistant": "I'm doing well!"}
```

**Fields**:
- `timestamp`: ISO 8601 format with microseconds
- `user`: User's message
- `assistant`: Model's generated response

**File Location**: `./chat_history.jsonl` (relative to app.py)

---

## Usage Examples

### JavaScript (Fetch API)

```javascript
// Send a chat message
async function sendMessage(message) {
  const response = await fetch('http://localhost:5000/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });

  const data = await response.json();
  console.log('Response:', data.response);
  console.log('Total chats:', data.chat_count);
}

// Get statistics
async function getStats() {
  const response = await fetch('http://localhost:5000/stats');
  const data = await response.json();
  console.log('Stats:', data);
}
```

### Python (requests)

```python
import requests

# Send a chat message
response = requests.post('http://localhost:5000/chat', json={
    'message': 'What is AI?'
})
data = response.json()
print(f"Response: {data['response']}")
print(f"Chat count: {data['chat_count']}")

# Get statistics
response = requests.get('http://localhost:5000/stats')
stats = response.json()
print(f"Ready for finetuning: {stats['ready_for_finetuning']}")
```

### cURL

```bash
# Send a chat message
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain neural networks"}'

# Get statistics
curl http://localhost:5000/stats
```

---

## Rate Limiting

**Not implemented**. The server can handle requests as fast as they arrive, limited only by:
- Model inference speed (~100ms-5s depending on device)
- Flask's single-threaded nature (development server)

For production, consider:
- Adding rate limiting middleware
- Using a production WSGI server (gunicorn, uwsgi)
- Implementing request queuing

---

## Authentication

**Not implemented**. All endpoints are publicly accessible.

For production, consider:
- API key authentication
- JWT tokens
- OAuth 2.0

---

## WebSocket Support

**Not supported**. All communication is HTTP request/response.

For streaming responses, consider:
- Server-Sent Events (SSE)
- WebSockets
- HTTP chunked transfer encoding

---

## API Versioning

**Not versioned**. All endpoints are at the root path.

For future-proofing, consider:
- `/api/v1/chat`
- `/api/v1/stats`

---

## Health Check

**Not implemented**. To add a health check endpoint:

```python
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'device': device,
        'model_loaded': model is not None
    })
```

---

## Monitoring & Metrics

**Not implemented**. Consider adding:
- Request count metrics
- Response time tracking
- Error rate monitoring
- Model performance metrics

---

## Development vs Production

### Current (Development)
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

- **Debug mode enabled**: Auto-reload on code changes
- **Not suitable for production**: Single-threaded, security risks

### Production Recommendations

Use a production WSGI server:

**Gunicorn**:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**uWSGI**:
```bash
uwsgi --http 0.0.0.0:5000 --wsgi-file app.py --callable app --processes 4
```

Disable debug mode in production:
```python
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```
