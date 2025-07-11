from fastapi import APIRouter, HTTPException, status
try:
    from OEM.utils.mongo import OEMConfigModel, get_oem_configs_collection
except ModuleNotFoundError:
    from utils.mongo import OEMConfigModel, get_oem_configs_collection
from typing import List

router = APIRouter(prefix="/oem/config", tags=["OEM Config"])

@router.post("/", response_model=OEMConfigModel, status_code=status.HTTP_201_CREATED)
async def create_oem_config(config: OEMConfigModel):
    collection = get_oem_configs_collection()
    existing = await collection.find_one({"product_model": config.product_model})
    if existing:
        raise HTTPException(status_code=400, detail="Config for this product_model already exists.")
    await collection.insert_one(config.dict())
    return config

@router.get("/{product_model}", response_model=OEMConfigModel)
async def get_oem_config(product_model: str):
    collection = get_oem_configs_collection()
    config = await collection.find_one({"product_model": product_model})
    if not config:
        raise HTTPException(status_code=404, detail="Config not found")
    config["id"] = str(config["_id"])  # Optionally expose MongoDB _id
    config.pop("_id", None)
    return OEMConfigModel(**config)

@router.put("/{product_model}", response_model=OEMConfigModel)
async def update_oem_config(product_model: str, update: OEMConfigModel):
    collection = get_oem_configs_collection()
    result = await collection.replace_one({"product_model": product_model}, update.dict())
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Config not found")
    return update

@router.get("/", response_model=List[OEMConfigModel])
async def list_oem_configs():
    collection = get_oem_configs_collection()
    configs = []
    async for config in collection.find():
        config["id"] = str(config["_id"])  # Optionally expose MongoDB _id
        config.pop("_id", None)
        configs.append(OEMConfigModel(**config))
    return configs 