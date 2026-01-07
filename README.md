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

## Notes

- The model parameter is currently set to `gpt-4o-2024-08-06`. When GPT-4.1 fine-tuning becomes available, update the model parameter in `main.py`.
- Make sure your JSONL file follows OpenAI's fine-tuning format with "messages" array containing "role" and "content" fields.

