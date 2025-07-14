import logging
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from typing import Optional

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/api/customerSupport")
async def customer_support(
    user_id: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    context: Optional[str] = Form(None)
):
    """
    Mock endpoint to forward a query to human support.
    """
    logger.info(f"Forwarding to human support: user_id={user_id}, query={query}, context={context}")
    # Here you could log to DB or send notification, but for now just mock
    return JSONResponse({
        "message": "Your request has been forwarded to a human agent. You’ll be contacted soon.",
        "user_id": user_id,
        "query": query,
        "context": context
    })
