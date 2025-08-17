
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