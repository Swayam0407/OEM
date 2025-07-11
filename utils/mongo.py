# utils/mongo.py
from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = "Test"

client = AsyncIOMotorClient(MONGODB_URL)
db = client[DB_NAME]

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
