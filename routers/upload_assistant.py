import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Body, Query
import re
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF
from tempfile import SpooledTemporaryFile
from pdf2image import convert_from_bytes
import pytesseract
import logging
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
from utils.mongo import get_assistants_collection, get_resources_collection
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
# Define assistantDetailSchemaProperties inline (Python version)
assistantDetailSchemaProperties = [
    {
        "name": "product_id",
        "dataType": ["text"],
        "description": "The product_id",
        "moduleConfig": {"text2vec-openai": {"skip": True}},
    },
    {
        "name": "groupId",
        "dataType": ["text"],
        "description": "The mongoDB Group",
        "moduleConfig": {"text2vec-openai": {"skip": True}},
    },
    {
        "name": "groupLabel",
        "dataType": ["text"],
        "description": "label of the group",
        "moduleConfig": {"text2vec-openai": {"skip": True}},
    },
    {
        "name": "detailId",
        "dataType": ["text"],
        "description": "objectId of the detail",
        "moduleConfig": {"text2vec-openai": {"skip": True}},
    },
    {
        "name": "detailLabel",
        "dataType": ["text"],
        "description": "Label of the detail",
        "moduleConfig": {"text2vec-openai": {"skip": True}},
    },
    {
        "name": "detailValue",
        "dataType": ["text"],
        "description": "Value of the Detail",
        "moduleConfig": {"text2vec-openai": {"skip": True}},
    },
]

load_dotenv()

router = APIRouter()

# OpenAI setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Weaviate setup
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")


def get_weaviate_client():
    try:
        auth_config = Auth.api_key(WEAVIATE_API_KEY)
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=auth_config,
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
            skip_init_checks=True  # Skip initial health checks to avoid connection issues
        )
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise HTTPException(status_code=503, detail=f"Weaviate connection failed: {str(e)}")


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text(text, max_length=1000):
    words = text.split()
    chunks, current, current_len = [], [], 0
    for word in words:
        if current_len + len(word) + 1 > max_length:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(word)
        current_len += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def embed_text_with_openai(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small", input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def ensure_simplifiedchunk_collection(client):
    try:
        if not client.collections.exists("SimplifiedChunk"):
            from weaviate.classes.config import Configure, Property, DataType
            client.collections.create(
                name="SimplifiedChunk",
                description="Simple chunks of PDF text with basic embeddings.",
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small"
                ),
                properties=[
                    Property(name="content", data_type=DataType.TEXT, description="The original text content"),
                    Property(name="source", data_type=DataType.TEXT, description="Source filename"),
                    Property(name="chunk_index", data_type=DataType.INT, description="Index of this chunk within the document"),
                    Property(name="document_id", data_type=DataType.TEXT, description="Unique identifier for the source document"),
                    Property(name="product_id", data_type=DataType.TEXT, description="Product identifier", skip_vectorization=True)
                ]
            )
            logger.info("Created collection: SimplifiedChunk")
        else:
            logger.info("Collection 'SimplifiedChunk' already exists.")
    except Exception as e:
        logger.error(f"Error ensuring SimplifiedChunk collection: {e}")


def store_chunk_in_weaviate(chunk_data, filename, chunk_index, document_id, product_id):
    try:
        client = get_weaviate_client()
        print(f"Connecting to Weaviate....", client.is_ready())
        try:
            ensure_simplifiedchunk_collection(client)
            properties = {
                "content": chunk_data["text"],
                "source": filename,
                "chunk_index": chunk_index,
                "document_id": document_id,
                "product_id": product_id
            }
            try:
                collection = client.collections.get("SimplifiedChunk")
                collection.data.insert(
                    properties=properties,
                    vector=chunk_data["embedding"]
                )
                return True
            except Exception as e:
                logger.error(f"Error inserting data: {e}")
                return str(e)
            finally:
                client.close()  # Always close the connection
        except Exception as e:
            logger.error(f"Error storing chunk in Weaviate: {e}")
            return str(e)
    except Exception as e:
        logger.error(f"Error connecting to Weaviate: {e}")
        return str(e)


# --- Assistant Schema Models ---
class AssistantPrompt(BaseModel):
    _id: str
    title: str
    prompt: str
    description: str

class AssistantLink(BaseModel):
    _id: str
    role: str  # "user" | "system"
    type: str
    label: str
    publicUrl: str
    privateUrl: str

class AssistantLinkGroup(BaseModel):
    _id: str
    label: str
    data: List[AssistantLink]

class AssistantDetail(BaseModel):
    _id: str
    type: str
    label: str
    value: Any

class AssistantDetailGroup(BaseModel):
    _id: str
    label: str
    data: List[AssistantDetail]

class AssistantUserForm(BaseModel):
    status: bool = False
    recaptcha: bool = False
    data: List[Dict[str, Any]] = []

class AssistantCarouselImage(BaseModel):
    _id: str
    url: str
    title: str

class AssistantModel(BaseModel):
    product_id: str
    name: str
    about: str
    voice: str = "alloy"
    avatar: str = "/images/defaultAssistantAvatar.jpg"
    isPublic: bool = False
    prompts: list = []
    links: list = []
    details: list = []
    userForm: dict = {}
    carouselImages: list = []
    createdAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updatedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def parse_json_field(field, default):
    try:
        return default if field is None else json.loads(field)
    except Exception:
        return default

# Add a dedicated parser for userForm to ensure it's always a dict

def parse_user_form(field):
    try:
        val = {"status": False, "recaptcha": False, "data": []} if field is None else json.loads(field)
        if not isinstance(val, dict):
            return {"status": False, "recaptcha": False, "data": []}
        return val
    except Exception:
        return {"status": False, "recaptcha": False, "data": []}



def validate_product_id(product_id: str) -> bool:
    # Only allow lowercase letters, numbers, underscores, and hyphens
    pattern = re.compile(r'^[a-z0-9_-]+$')
    return bool(pattern.match(product_id))


@router.post(
    "/api/uploadAssistant", 
    description="Upload a new assistant with product documentation",
    response_description="Returns the uploaded assistant details and processing status",
    response_model=dict,
    status_code=200,
)
async def upload_assistant(
    pdf: UploadFile = File(..., description="PDF file containing product documentation"),
    product_id: str = Form(..., description="Unique identifier for the product (e.g., samsung_eco_7kg)"),
    name: str = Form(..., description="Name of the assistant"),
    about: str = Form(..., description="Description of the assistant"),
    voice: Optional[str] = Form(None),
    avatar: Optional[str] = Form(None),
    isPublic: Optional[str] = Form(None),
    prompts: Optional[str] = Form(None),
    links: Optional[str] = Form(None),
    details: Optional[str] = Form(None),
    userForm: Optional[str] = Form(None),
    carouselImages: Optional[str] = Form(None),
):
    logger.info(f"Received upload: {pdf.filename}, name: {name}, about: {about}, voice: {voice}, isPublic: {isPublic}, product_id: {product_id}")
    # Validate product_id
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")
    if not validate_product_id(product_id):
        raise HTTPException(status_code=400, detail="Invalid product_id format. Use only lowercase letters, numbers, underscores, and hyphens")

    # Parse booleans and JSON fields
    is_public = (isPublic.lower() == "true") if isPublic else False
    prompts_val = parse_json_field(prompts, [])
    links_val = parse_json_field(links, [])
    details_val = parse_json_field(details, [])
    userForm_val = parse_user_form(userForm)
    carouselImages_val = parse_json_field(carouselImages, [])
    # Extract text from PDF (existing logic)
    with SpooledTemporaryFile() as tmp:
        content = await pdf.read()
        tmp.write(content)
        tmp.seek(0)
        doc = fitz.open(stream=tmp.read(), filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
    # OCR image captions (existing logic)
    image_captions = []
    try:
        images = convert_from_bytes(content)
        for idx, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            if text.strip():
                image_captions.append(text.strip())
            logger.info(f"OCR done for image {idx+1}/{len(images)}.")
    except Exception as e:
        logger.warning(f"Image OCR failed: {e}")
        image_captions.append(f"OCR error: {str(e)}")
    if image_captions:
        full_text += "\n\n" + "\n".join(image_captions)
    text_chunks = chunk_text(full_text)
    logger.info(f"Chunked text into {len(text_chunks)} parts.")
    upload_errors = []
    num_uploaded = 0
    document_id = f"{pdf.filename}_{hash(name)}"  # Unique ID per document+name
    assistants_collection = get_assistants_collection()
    # Build the assistant document using the model
    assistant_doc = AssistantModel(
        product_id=product_id,
        name=name,
        about=about,
        voice=voice or "alloy",
        avatar=avatar or "/images/defaultAssistantAvatar.jpg",
        isPublic=is_public,
        prompts=prompts_val,
        links=links_val,
        details=details_val,
        userForm=userForm_val,
        carouselImages=carouselImages_val,
        createdAt=datetime.now(timezone.utc),
        updatedAt=datetime.now(timezone.utc),
    ).model_dump()
    # Insert into MongoDB
    await assistants_collection.insert_one(assistant_doc)

    # --- Insert resource document for uploaded file ---
    resources_collection = get_resources_collection()
    file_extension = os.path.splitext(pdf.filename)[1][1:] if '.' in pdf.filename else ''
    resource_doc = {
        "product_id": product_id,  # Primary identifier
        "title": pdf.filename,
        "url": f"/uploads/{pdf.filename}",  # Adjust if using S3 or other storage
        "size": len(content),
        "type": "document",
        "extension": file_extension,
        "aiAccess": False,
        "aiIngestion": False,
        "isPublic": False,
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
    }
    await resources_collection.insert_one(resource_doc)

    # --- Create empty Weaviate details collection for this assistant ---
    try:
        w_client = get_weaviate_client()
        details_collection_name = f"Product_Detail_{product_id}"
        if not w_client.collections.exists(details_collection_name):
            from weaviate.classes.config import Configure, Property, DataType
            w_client.collections.create(
                name=details_collection_name,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
                properties=[
                    Property(name="product_id", data_type=DataType.TEXT, description="The product_id", skip_vectorization=True),
                    Property(name="groupId", data_type=DataType.TEXT, description="The mongoDB Group", skip_vectorization=True),
                    Property(name="groupLabel", data_type=DataType.TEXT, description="label of the group", skip_vectorization=True),
                    Property(name="detailId", data_type=DataType.TEXT, description="objectId of the detail", skip_vectorization=True),
                    Property(name="detailLabel", data_type=DataType.TEXT, description="Label of the detail", skip_vectorization=True),
                    Property(name="detailValue", data_type=DataType.TEXT, description="Value of the Detail", skip_vectorization=True),
                ]
            )
        w_client.close()  # Close the connection after use
    except Exception as e:
        logger.error(f"Failed to create Weaviate collection: {e}")
        upload_errors.append(f"Weaviate collection creation failed: {str(e)}")
    # Generate embeddings and store chunks
    for i, chunk in enumerate(text_chunks):
        try:
            embedding = embed_text_with_openai(chunk)
            if embedding is None:
                raise Exception("Failed to generate embedding")
            chunk_data = {
                "text": chunk,
                "embedding": embedding
            }
            result = store_chunk_in_weaviate(chunk_data, pdf.filename, i, document_id, product_id)
            if result is True:
                num_uploaded += 1
                logger.info(f"Stored chunk {i+1} in Weaviate")
            else:
                upload_errors.append(f"Failed to store chunk {i+1}: {result}")
        except Exception as e:
            error_msg = f"Error processing chunk {i+1}: {str(e)}"
            logger.error(error_msg)
            upload_errors.append(error_msg)
    return JSONResponse(
        {
            "message": "Completed simplified upload processing.",
            "filename": pdf.filename,
            "name": name,
            "about": about,
            "voice": voice or "alloy",
            "avatar": avatar or "/images/defaultAssistantAvatar.jpg",
            "isPublic": is_public,
            "prompts": prompts_val,
            "links": links_val,
            "details": details_val,
            "userForm": userForm_val,
            "carouselImages": carouselImages_val,
            "document_id": document_id,
            "num_chunks": len(text_chunks),
            "num_uploaded": num_uploaded,
            "upload_errors": upload_errors,
            "image_caption_sample": image_captions[:2],
        }
    )
