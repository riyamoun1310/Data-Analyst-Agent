from fastapi import File, UploadFile
import pandas as pd
import io
from utils.validation import validate_weather_columns, validate_sales_columns
from services.weather_service import analyze_weather
from services.sales_service import analyze_sales
# POST / endpoint for compatibility with test harnesses that POST to root
@app.post("/")
async def root_post(file: UploadFile = File(...)):
    try:
        content = await file.read()
        csv_string = content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_string))
        if validate_weather_columns(df):
            return analyze_weather(df)
        elif validate_sales_columns(df):
            return analyze_sales(df)
        else:
            return {"error": "Unrecognized CSV format. Only weather and sales datasets are supported."}
    except Exception as e:
        return {"error": str(e)}

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.analyze_router import router as analyze_router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data-analyst-agent")

app = FastAPI(
    title="Data Analyst Agent API",
    description="API for automated data analysis and visualization",
    version="2.0.0"
)

# CORS for all origins (change for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze_router)


@app.get("/")
async def root():
    return {
        "message": "Data Analyst Agent API is running",
        "status": "healthy",
        "version": "2.0.0",
        "endpoints": [
            "POST /api/ - Main analysis endpoint (CSV upload)",
            "GET /health - Health check"
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Service is up"
    }

# Entrypoint for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)