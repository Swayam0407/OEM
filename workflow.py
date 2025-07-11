from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import logging
from openai import OpenAI
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any, Optional
from utils.mongo import get_assistants_collection
from bson import ObjectId

load_dotenv()

router = APIRouter()

# OpenAI setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Weaviate setup
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")


def get_weaviate_client():
    auth_config = Auth.api_key(WEAVIATE_API_KEY)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=auth_config,
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
    )
    return client


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    instruction: str
    document_id: Optional[str] = None  # Optional: search specific document
    assistant_id: str  # Now required


def extract_topics_from_instruction(instruction: str) -> List[str]:
    """
    Use LLM to analyze OEM instruction and extract specific topics
    """
    prompt = f"""
Analyze this instruction and extract the key topics the user wants to find. 
Return ONLY a JSON array of topic strings, nothing else.

Instruction: {instruction}

Examples:
- "Extract safety precautions and warranty details" → ["safety precautions", "warranty details"]
- "Find installation procedures and troubleshooting steps" → ["installation procedures", "troubleshooting steps"]  
- "Get technical specifications and maintenance guidelines" → ["technical specifications", "maintenance guidelines"]

Your response (JSON array only):"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        
        # Clean the content - remove any markdown formatting
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        logger.info(f"GPT raw response: {content}")
        
        # Parse JSON response
        topics = json.loads(content)
        if isinstance(topics, list) and len(topics) > 0:
            logger.info(f"Successfully extracted topics: {topics}")
            return topics
        else:
            raise ValueError("Response is not a valid list or is empty")
            
    except Exception as e:
        logger.error(f"Topic extraction error: {e}")
        # Better fallback: extract meaningful phrases
        import re
        # Look for quoted phrases or key terms
        quoted_phrases = re.findall(r'"([^"]*)"', instruction)
        if quoted_phrases:
            return quoted_phrases[:3]
            
        # Extract noun phrases (simple approach)
        words = instruction.lower().split()
        if "safety" in words and "precautions" in words:
            topics = ["safety precautions"]
        else:
            topics = ["safety"]
            
        if "warranty" in words:
            topics.append("warranty details")
            
        return topics if topics else ["general information"]


def hybrid_search_for_topic(topic: str, document_id: str = None, limit: int = 5) -> List[Dict]:
    """
    Perform hybrid search for a specific topic using text-embedding-3-small
    """
    try:
        logger.info(f"[DEBUG] Entered hybrid_search_for_topic for topic: {topic}, document_id: {document_id}, limit: {limit}")
        client_weaviate = get_weaviate_client()
        logger.info(f"[DEBUG] Weaviate client obtained. Checking if collection exists...")
        
        # Check if collection exists
        if not client_weaviate.collections.exists("SimplifiedChunk"):
            logger.error("SimplifiedChunk collection does not exist")
            return []
        logger.info(f"[DEBUG] Collection exists. Preparing query...")
        
        # Use hybrid search with text-embedding-3-small
        query = {
            "hybrid": {
                "query": topic,
                "alpha": 0.3
            },
            "limit": limit
        }
        if document_id:
            query["where"] = {
                "path": ["document_id"],
                "operator": "Equal",
                "valueText": document_id
            }
        result = client_weaviate.collections.get("SimplifiedChunk").query.hybrid(**query)
        logger.info(f"[DEBUG] Weaviate query executed. Raw result: {result}")
        
        chunks = []
        for item in result.objects:
            chunks.append({
                "content": item.properties.get("content", ""),
                "source": item.properties.get("source", ""),
                "chunk_index": item.properties.get("chunk_index", 0),
                "document_id": item.properties.get("document_id", "")
            })
        logger.info(f"[DEBUG] Found {len(chunks)} chunks for topic: {topic}")
        return chunks
        
    except Exception as e:
        logger.error(f"[DEBUG] Search error for topic '{topic}': {e}")
        logger.error(f"[DEBUG] Error type: {type(e)}")
        import traceback
        logger.error(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return []


def generate_final_json(topics_with_chunks: Dict[str, List[Dict]]) -> Dict[str, str]:
    """
    Use LLM to process all search results and generate final structured JSON
    """
    # Check if we have any chunks at all
    total_chunks = sum(len(chunks) for chunks in topics_with_chunks.values())
    if total_chunks == 0:
        logger.warning("No chunks found for any topic")
        return {topic: "No relevant information found in the document" for topic in topics_with_chunks.keys()}
    
    # Prepare context for LLM
    context_parts = []
    for topic, chunks in topics_with_chunks.items():
        context_parts.append(f"\n=== TOPIC: {topic} ===")
        if not chunks:
            context_parts.append("No relevant chunks found for this topic.")
        else:
            for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks per topic
                chunk_text = chunk['content'][:800] if len(chunk['content']) > 800 else chunk['content']  # Limit chunk size
                context_parts.append(f"Chunk {i+1}: {chunk_text}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""
Based on the search results below, extract and summarize the information for each topic.
Return ONLY a valid JSON object. Each key should be the topic name and the value should be the extracted content.

If no relevant information is found for a topic, set the value to "No specific information found".

Search Results:
{context}

Return JSON only (no markdown, no extra text):"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()
        
        # Clean the content - remove any markdown formatting
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        logger.info(f"GPT final JSON raw response: {content}")
        
        # Parse JSON response
        result = json.loads(content)
        if isinstance(result, dict):
            logger.info(f"Successfully generated final JSON: {result}")
            return result
        else:
            raise ValueError("Response is not a JSON object")
            
    except Exception as e:
        logger.error(f"Final JSON generation error: {e}")
        # Fallback: create simple structure from the chunks we have
        fallback_result = {}
        for topic, chunks in topics_with_chunks.items():
            if chunks:
                # Extract first chunk content as fallback
                content_preview = chunks[0]['content'][:200] + "..." if len(chunks[0]['content']) > 200 else chunks[0]['content']
                fallback_result[topic] = f"Found {len(chunks)} relevant section(s). Preview: {content_preview}"
            else:
                fallback_result[topic] = "No relevant information found for this topic"
        
        return fallback_result


@router.post("/search-workflow")
async def search_workflow(request: SearchRequest):
    """
    Main workflow endpoint:
    1. Extract topics from instruction
    2. Perform hybrid search for each topic  
    3. Generate final structured JSON
    4. Update assistant details in MongoDB
    """
    logger.info(f"Starting search workflow for instruction: {request.instruction}")
    try:
        # Step 1: Extract topics
        topics = extract_topics_from_instruction(request.instruction)
        logger.info(f"Extracted topics: {topics}")
        if not topics:
            raise HTTPException(status_code=400, detail="Could not extract topics from instruction")
        # Step 2: Hybrid search for each topic
        topics_with_chunks = {}
        for topic in topics:
            chunks = hybrid_search_for_topic(topic, request.document_id)
            topics_with_chunks[topic] = chunks
            logger.info("Weaviate search for topic '%s' returned %d chunks", topic, len(chunks))
        # Step 3: Generate final JSON
        final_result = generate_final_json(topics_with_chunks)
        logger.info("OpenAI summarization complete")
        # Step 4: Update assistant details in MongoDB
        assistants_collection = get_assistants_collection()
        details = [
            {
                "label": topic,
                "data": [
                    {
                        "label": topic,
                        "value": value,
                        "_id": str(ObjectId())
                    }
                ],
                "_id": str(ObjectId())
            }
            for topic, value in final_result.items()
        ]
        await assistants_collection.update_one(
            {"_id": ObjectId(request.assistant_id)},
            {"$set": {"details": details}}
        )

        # --- Update Weaviate details collection ---
        WEAVIATE_URL = os.getenv("WEAVIATE_URL")
        WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        w_client = get_weaviate_client()
        details_collection_name = f"Assistant_Detail_{request.assistant_id}"
        # Create collection if it doesn't exist
        if not w_client.collections.exists(details_collection_name):
            from weaviate.classes.config import Configure
            w_client.collections.create(
                name=details_collection_name,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
                properties=[
                    Configure.Property(name="assistantId", data_type=Configure.DataType.TEXT, description="The mongoDB assistantId", skip_vectorization=True),
                    Configure.Property(name="groupId", data_type=Configure.DataType.TEXT, description="The mongoDB Group", skip_vectorization=True),
                    Configure.Property(name="groupLabel", data_type=Configure.DataType.TEXT, description="label of the group", skip_vectorization=True),
                    Configure.Property(name="detailId", data_type=Configure.DataType.TEXT, description="objectId of the detail", skip_vectorization=True),
                    Configure.Property(name="detailLabel", data_type=Configure.DataType.TEXT, description="Label of the detail", skip_vectorization=True),
                    Configure.Property(name="detailValue", data_type=Configure.DataType.TEXT, description="Value of the Detail", skip_vectorization=True),
                ]
            )
        # Remove all previous details for this assistant (optional: for full sync)
        w_client.collections.get(details_collection_name).objects.delete_many(
            where={
                "path": ["assistantId"],
                "operator": "Equal",
                "valueText": str(request.assistant_id)
            }
        )
        # Insert new details
        for group in details:
            for item in group["data"]:
                w_client.collections.get(details_collection_name).objects.create(
                    properties={
                        "assistantId": str(request.assistant_id),
                        "groupId": group["_id"],
                        "groupLabel": group["label"],
                        "detailId": item["_id"],
                        "detailLabel": item["label"],
                        "detailValue": str(item["value"]),
                    }
                )
        return JSONResponse({
            "instruction": request.instruction,
            "document_id": request.document_id,
            "assistant_id": request.assistant_id,
            "extracted_topics": topics,
            "results": final_result,
            "metadata": {
                "total_topics": len(topics),
                "total_chunks_found": sum(len(chunks) for chunks in topics_with_chunks.values())
            }
        })
    except Exception as e:
        logger.error(f"Search workflow error: {e}")
        raise HTTPException(status_code=500, detail=f"Search workflow failed: {str(e)}")
