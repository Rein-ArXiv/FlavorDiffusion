from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn
import io
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def main_page():
    return "This is FlavorDiffusion model API"


current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "../dataset/input_data.json")

# JSON 파일 로드 (서버 시작 시 한 번만 실행)
with open(dataset_path, "r", encoding="utf-8") as f:
    input_data = json.load(f)

@app.get("/test")
def test_page():
    return "아니 왜 작동을 안해"

@app.get("/input_items")
async def get_items(
    start: int = Query(0, alias="start"),
    limit: int = Query(100, alias="limit"),
    search: str = Query(None, alias="search")  # 검색어 추가
):
    """ 요청된 start부터 limit 개수만큼 데이터를 반환하는 API (검색 포함) """
    
    # 검색어가 있다면 필터링
    filtered_data = [item for item in input_data if search.lower() in item["name"].lower()] if search else input_data
    
    end = start + limit
    return {"data": filtered_data[start:end], "total": len(filtered_data)}
