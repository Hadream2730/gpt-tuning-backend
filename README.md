# GPT Fine-Tuning Backend

FastAPI backend for fine-tuning GPT-4.1 models using OpenAI's API.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

3. Run the server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `POST /api/fine-tune` - Upload JSONL file and start fine-tuning
- `GET /api/fine-tune/{job_id}` - Get status of a fine-tuning job

## Deployment on Render.com

1. **Start Command:**
   ```
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
   Or Render will automatically use the `Procfile` if present.

2. **Environment Variables to set in Render:**
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `ALLOWED_ORIGINS` - Comma-separated list of frontend URLs (e.g., `https://your-frontend.onrender.com,http://localhost:3000`)

3. **Build Command:**
   ```
   pip install -r requirements.txt
   ```

4. **Python Version:**
   Set to Python 3.11 or 3.12 in Render settings.

## Notes

- The model parameter is currently set to `gpt-4o-2024-08-06`. When GPT-4.1 fine-tuning becomes available, update the model parameter in `main.py`.
- Make sure your JSONL file follows OpenAI's fine-tuning format with "messages" array containing "role" and "content" fields.
- For production, update the `ALLOWED_ORIGINS` environment variable to include your frontend URL.

