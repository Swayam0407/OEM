from fastapi import APIRouter, HTTPException, status
try:
    from OEM.utils.mongo import SessionStateModel, get_session_states_collection
except ModuleNotFoundError:
    from utils.mongo import SessionStateModel, get_session_states_collection
from typing import List

router = APIRouter(prefix="/session/state", tags=["Session State"])

@router.post("/", response_model=SessionStateModel, status_code=status.HTTP_201_CREATED)
async def create_session_state(state: SessionStateModel):
    collection = get_session_states_collection()
    existing = await collection.find_one({"session_id": state.session_id})
    if existing:
        raise HTTPException(status_code=400, detail="Session state for this session_id already exists.")
    await collection.insert_one(state.dict())
    return state

@router.get("/{session_id}", response_model=SessionStateModel)
async def get_session_state(session_id: str):
    collection = get_session_states_collection()
    state = await collection.find_one({"session_id": session_id})
    if not state:
        raise HTTPException(status_code=404, detail="Session state not found")
    state["id"] = str(state["_id"])  # Optionally expose MongoDB _id
    state.pop("_id", None)
    return SessionStateModel(**state)

@router.put("/{session_id}", response_model=SessionStateModel)
async def update_session_state(session_id: str, update: SessionStateModel):
    collection = get_session_states_collection()
    result = await collection.replace_one({"session_id": session_id}, update.dict())
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Session state not found")
    return update

@router.get("/", response_model=List[SessionStateModel])
async def list_session_states():
    collection = get_session_states_collection()
    states = []
    async for state in collection.find():
        state["id"] = str(state["_id"])  # Optionally expose MongoDB _id
        state.pop("_id", None)
        states.append(SessionStateModel(**state))
    return states 