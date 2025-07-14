import logging
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
from typing import Optional

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/api/raiseTicket")
async def raise_ticket(
    user_id: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    contact: Optional[str] = Form(None),
    description: str = Form(...),
    priority: Optional[str] = Form("medium"),
    product: Optional[str] = Form(None),
    agent_context: Optional[str] = Form(None),
    attachments: Optional[str] = Form(None)
):
    """
    Endpoint to raise a support ticket for unresolved queries.
    """
    from utils.mongo import get_database
    db = get_database()
    tickets = db["tickets"]
    ticket_doc = {
        "user_id": user_id,
        "name": name,
        "contact": contact,
        "description": description,
        "priority": priority,
        "product": product,
        "agent_context": agent_context,
        "attachments": attachments,
        "createdAt": datetime.now(timezone.utc),
        "status": "open"
    }
    result = await tickets.insert_one(ticket_doc)
    ticket_id = str(result.inserted_id)
    logger.info(f"Ticket raised: {ticket_id}")
    return JSONResponse({
        "message": "Ticket raised successfully.",
        "ticket_id": ticket_id,
        "status": "open"
    })
