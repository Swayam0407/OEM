# utils/mongo.py
from motor.motor_asyncio import AsyncIOMotorClient
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = "Test"

client = AsyncIOMotorClient(MONGODB_URL)
db = client[DB_NAME]

class AskFlowItemModel(BaseModel):
    field: str
    priority: Literal["high", "medium", "low"]
    preferred_stage: Literal["before_escalation", "after_troubleshooting"]
    reason: str

class EscalationPolicyModel(BaseModel):
    threshold_fail_steps: int

class OEMConfigModel(BaseModel):
    product_model: str
    oem: str
    ask_flow: List[AskFlowItemModel]
    escalation_policy: EscalationPolicyModel
    tools_allowed: List[str]
    voice_tone: Literal["friendly", "formal"]
    language: str

class SessionStateModel(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    product_model: str
    step_count: int
    fields_collected: Dict[str, str]  # e.g., { 'serial_number': '1234', 'location': '' }

class RAGChunkModel(BaseModel):
    product_model: str
    content: str
    section: Optional[str] = None  # e.g., manual section or page reference

def get_workflows_collection():
    return db["workflows"]

def get_workflow_runs_collection():
    return db["workflow_runs"]

def get_assistants_collection():
    return db["assistants"]

def get_resources_collection():
    return db["resources"]

def get_database():
    return db

def get_oem_configs_collection():
    return db["oem_configs"]

def get_session_states_collection():
    return db["session_states"]

def get_rag_chunks_collection():
    return db["rag_chunks"]
