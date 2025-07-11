from pathlib import Path
from dotenv import load_dotenv

# Always load the .env residing next to this file so that running the
# app from any working directory still picks up environment variables.
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import workflow, upload_assistant
try:
    from OEM.routers import oem_config, session_state  # When run as package (e.g., uvicorn OEM.main:app)
except ModuleNotFoundError:
    from routers import oem_config, session_state  # When run inside the OEM directory (e.g., uvicorn main:app)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflow.router, prefix="/api")
app.include_router(upload_assistant.router)
app.include_router(oem_config.router)
app.include_router(session_state.router)


@app.get("/")
async def root():
    return {"message": "AI Workflow Builder API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)
