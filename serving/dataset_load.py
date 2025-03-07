from fastapi import FastAPI, UploadFile, File, Query
import pandas as pd
import numpy as np
import joblib
import uvicorn
import io
import json

app = FastAPI()

# ML 모델 로드 (예제)
# model = joblib.load("model.pkl")  # 학습된 모델 로드

@app.get("/")
def main_page():
    return "This is FlavorDiffusion model API"


# JSON 파일 로드 (서버 시작 시 한 번만 실행)
with open("../dataset/input_data.json", "r", encoding="utf-8") as f:
    input_data = json.load(f)

@app.get("/test")
def test_page():
    return "아니 왜 작동을 안해"

@app.get("/items")
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
