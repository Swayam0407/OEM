from fastapi import APIRouter, HTTPException, status
try:
    from OEM.utils.mongo import OEMConfigModel, get_oem_configs_collection
except ModuleNotFoundError:
    from utils.mongo import OEMConfigModel, get_oem_configs_collection

router = APIRouter(prefix="/oem/config", tags=["OEM Config"])

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_oem_config(config: OEMConfigModel):
    collection = get_oem_configs_collection()
    # Use product_id as the key for consistency
    existing = await collection.find_one({"product_id": config.product_id})
    if existing:
        raise HTTPException(status_code=400, detail="Config for this product_id already exists.")
    result = await collection.insert_one(config.dict())
    config_id = str(result.inserted_id)
    response = config.dict()
    response["config_id"] = config_id
    return response

@router.put("/{product_id}")
async def update_oem_config(product_id: str, update: OEMConfigModel):
    collection = get_oem_configs_collection()
    result = await collection.replace_one({"product_id": product_id}, update.dict())
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Config not found")
    updated_config = await collection.find_one({"product_id": product_id})
    response = update.dict()
    response["config_id"] = str(updated_config["_id"]) if updated_config and "_id" in updated_config else None
    return response