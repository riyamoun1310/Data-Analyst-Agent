from fastapi import APIRouter, File, UploadFile
import pandas as pd
import io
from utils.validation import validate_weather_columns, validate_sales_columns
from services.weather_service import analyze_weather
from services.sales_service import analyze_sales
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("data-analyst-agent")

router = APIRouter()

@router.post("/api/")
async def analyze_api(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        content = await file.read()
        csv_string = content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_string))
        cols = [c.lower() for c in df.columns]
        if validate_weather_columns(df):
            return JSONResponse(content=analyze_weather(df))
        elif validate_sales_columns(df):
            return JSONResponse(content=analyze_sales(df))
        else:
            return JSONResponse(
                content={"error": "Unrecognized CSV format. Only weather and sales datasets are supported."},
                status_code=400
            )
    except Exception as e:
        logger.error(f"Error in /api/: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
