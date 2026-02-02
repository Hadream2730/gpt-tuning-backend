import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

app = FastAPI(title="GPT Fine-Tuning API", version="1.0.0")

# CORS middleware to allow frontend to connect
# Get allowed origins from environment variable or use defaults
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=openai_api_key)


def validate_jsonl_file(file_path: str) -> bool:
    """Validate that the file is a proper JSONL file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) == 0:
                print("line: 0")
                return False
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Check if it has the required structure for fine-tuning
                    if not isinstance(data, dict):
                        print(f"Error: Line {i} is not a JSON object: {line}")
                        return False
                    if "messages" not in data:
                        print(f"Error: Line {i} does not have 'messages' field: {line}")
                        return False
                    if not isinstance(data["messages"], list):
                        print(f"Error: Line {i} 'messages' is not a list: {line}")
                        return False
                    # Validate message structure
                    for msg in data["messages"]:
                        if not isinstance(msg, dict):
                            print(f"Error: Line {i} message is not a JSON object: {msg}")
                            return False
                        if "role" not in msg or "content" not in msg:
                            print(f"Error: Line {i} message does not have 'role' or 'content' fields: {msg}")
                            return False
                except json.JSONDecodeError:
                    print(f"Error: Line {i} is not a valid JSON object: {line}")
                    return False
        return True
    except Exception as e:
        print(f"Error validating file: {e}")
        return False


@app.get("/")
async def root():
    return {"message": "GPT Fine-Tuning API is running"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/fine-tune")
async def fine_tune(file: UploadFile = File(...)):
    """
    Upload a JSONL file and start fine-tuning GPT-4.1 model.
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.jsonl', '.json')):
            raise HTTPException(
                status_code=400,
                detail="File must be in JSONL or JSON format"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.jsonl') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Validate JSONL format
            if not validate_jsonl_file(tmp_file_path):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSONL format. Each line must be a JSON object with 'messages' array containing 'role' and 'content' fields."
                )

            # Clean the file: remove empty lines and trailing newlines
            # Create a cleaned version for OpenAI upload
            cleaned_file_path = tmp_file_path + "_cleaned"
            valid_lines = []
            with open(tmp_file_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    stripped_line = line.strip()
                    if stripped_line:  # Only keep non-empty lines
                        valid_lines.append(stripped_line)
            
            # Write cleaned file without trailing newline
            with open(cleaned_file_path, 'w', encoding='utf-8') as f_out:
                if valid_lines:
                    f_out.write('\n'.join(valid_lines))

            # Upload cleaned file to OpenAI
            with open(cleaned_file_path, 'rb') as training_file:
                uploaded_file = client.files.create(
                    file=training_file,
                    purpose='fine-tune'
                )
            
            # Clean up cleaned file
            if os.path.exists(cleaned_file_path):
                os.unlink(cleaned_file_path)

            # Start fine-tuning job
            # Note: GPT-4.1 fine-tuning may not be available yet. 
            # Using gpt-4o-2024-08-06 or gpt-4-turbo as base model.
            # Adjust the model parameter based on OpenAI's available models.
            fine_tune_job = client.fine_tuning.jobs.create(
                training_file=uploaded_file.id,
                model="gpt-4.1-2025-04-14",  # Use available GPT-4 model. Change to "gpt-4.1" when available
                # hyperparameters={
                #     "n_epochs": 3,  # Optional: customize hyperparameters
                # }
            )

            return JSONResponse(
                status_code=200,
                content={
                    "message": "Fine-tuning job started successfully",
                    "job_id": fine_tune_job.id,
                    "status": fine_tune_job.status,
                    "model": fine_tune_job.model,
                    "created_at": fine_tune_job.created_at
                }
            )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        error_message = str(e)
        print(f"Error during fine-tuning: {error_message}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start fine-tuning: {error_message}"
        )


@app.get("/api/fine-tune/{job_id}")
async def get_fine_tune_status(job_id: str):
    """
    Get the status of a fine-tuning job.
    """
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        return {
            "job_id": job.id,
            "status": job.status,
            "model": job.model,
            "created_at": job.created_at,
            "finished_at": job.finished_at,
            "trained_tokens": job.trained_tokens,
            "error": job.error if hasattr(job, 'error') else None
        }
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Fine-tuning job not found: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

