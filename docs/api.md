# API

Base URL: `http://127.0.0.1:5000`

## Endpoints

### GET /

Serves `templates/index.html`.

### POST /chat

Send a message. Receive a generated response.

**Request**: `application/json`
```json
{"message": "string"}
```

**Response** (200):
```json
{"response": "string", "chat_count": number, "chat_id": "string (UUID)"}
```

**Errors**: 400 if message is missing, not a string, empty, or exceeds 1000 characters.

Generation parameters (hardcoded in `app.py`): `max_tokens=150`, `temperature=0.8`, `top_k=200`.

Rate limit: 20 requests/minute per IP.

### POST /rate

Rate a previous chat response (thumbs up/down).

**Request**: `application/json`
```json
{"chat_id": "string (UUID)", "rating": 0 | 1}
```

`rating`: `1` = thumbs up, `0` = thumbs down.

**Response** (200):
```json
{"status": "ok", "chat_id": "string", "rating": 0 | 1}
```

**Errors**: 400 if `chat_id` missing or `rating` invalid; 404 if `chat_id` not found.

Rate limit: 30 requests/minute per IP.

### GET /stats

```json
{"chat_count": number, "ready_for_finetuning": boolean}
```

`ready_for_finetuning` is true when `chat_count >= 10`.

## Security

| Measure | Detail |
|---------|--------|
| Rate limiting | 20/min on `/chat`, 30/min on `/rate`, 100/min default (Flask-Limiter, in-memory) |
| CORS | `127.0.0.1:5000` and `localhost:5000` only |
| Input validation | JSON required, string type, 1000 char limit, no whitespace-only |
| Binding | `127.0.0.1` — not reachable from external networks |
| Debug | Disabled |
| Headers | `X-Frame-Options: DENY`, `CSP: default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin` |
| Error handling | Catch-all returns generic 500, logs internally |

## Production

The dev server is single-threaded. For production, use a WSGI server:

```bash
gunicorn -w 4 -b 127.0.0.1:5000 app:app
```
