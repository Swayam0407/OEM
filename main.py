from pathlib import Path
from dotenv import load_dotenv

# Always load the .env residing next to this file so that running the
# app from any working directory still picks up environment variables.
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import workflow, upload_assistant

app = FastAPI(title="AI Workflow Builder", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflow.router, prefix="/api")
app.include_router(upload_assistant.router)


@app.get("/")
async def root():
    return {"message": "AI Workflow Builder API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
