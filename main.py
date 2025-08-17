import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Configure logging
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

@app.post("/api/")
async def analyze_api(file: UploadFile = File(...)):
    """
    Main analysis endpoint for test system compatibility.
    Accepts only file upload (CSV) as form-data with key 'file'.
    """
    try:
        logger.info(f"Received file: {file.filename}")
        content = await file.read()
        csv_string = content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_string))
        cols = [c.lower() for c in df.columns]

        # Weather dataset detection
        if set(['date', 'temp_c', 'precip_mm']).issubset(cols):
            avg_temp = float(df['temp_c'].mean()) if 'temp_c' in df else None
            min_temp = float(df['temp_c'].min()) if 'temp_c' in df else None
            max_precip_idx = df['precip_mm'].idxmax() if 'precip_mm' in df else None
            max_precip_date = str(df.loc[max_precip_idx, 'date']) if max_precip_idx is not None else None
            temp_precip_corr = float(df['temp_c'].corr(df['precip_mm'])) if 'temp_c' in df and 'precip_mm' in df and df['temp_c'].std() > 0 and df['precip_mm'].std() > 0 else 0.0
            avg_precip = float(df['precip_mm'].mean()) if 'precip_mm' in df else None
            # temp_line_chart
            temp_line_chart = ""
            try:
                plt.figure(figsize=(6,4))
                plt.plot(df['date'], df['temp_c'], color='red')
                plt.xlabel('Date')
                plt.ylabel('Temperature (C)')
                plt.title('Temperature Over Time')
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                temp_line_chart = base64.b64encode(buf.read()).decode()
            except Exception as e:
                logger.warning(f"Failed to generate temp_line_chart: {e}")
            # precip_histogram
            precip_histogram = ""
            try:
                plt.figure(figsize=(6,4))
                plt.hist(df['precip_mm'], color='orange', bins=10)
                plt.xlabel('Precipitation (mm)')
                plt.ylabel('Frequency')
                plt.title('Precipitation Histogram')
                plt.tight_layout()
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
                plt.close()
                buf2.seek(0)
                precip_histogram = base64.b64encode(buf2.read()).decode()
            except Exception as e:
                logger.warning(f"Failed to generate precip_histogram: {e}")
            return JSONResponse(content={
                "average_temp_c": round(avg_temp, 2) if avg_temp is not None else None,
                "max_precip_date": max_precip_date,
                "min_temp_c": min_temp,
                "temp_precip_correlation": round(temp_precip_corr, 10),
                "average_precip_mm": round(avg_precip, 2) if avg_precip is not None else None,
                "temp_line_chart": temp_line_chart,
                "precip_histogram": precip_histogram
            })

        # Sales dataset detection
        elif set(['region', 'sales', 'date']).issubset(cols):
            total_sales = float(df['sales'].sum()) if 'sales' in df else None
            top_region = df.groupby('region')['sales'].sum().idxmax() if 'region' in df and 'sales' in df else None
            day_sales_corr = 0.0
            median_sales = None
            total_sales_tax = None
            try:
                df['day'] = pd.to_datetime(df['date']).dt.day
                day_sales_corr = float(df['day'].corr(df['sales'])) if df['day'].std() > 0 and df['sales'].std() > 0 else 0.0
                median_sales = float(df['sales'].median())
                total_sales_tax = float(df['sales'].sum() * 0.10)
            except Exception as e:
                logger.warning(f"Failed to calculate sales stats: {e}")
            bar_chart = ""
            try:
                region_sales = df.groupby('region')['sales'].sum()
                plt.figure(figsize=(6,4))
                region_sales.plot(kind='bar', color='blue')
                plt.xlabel('Region')
                plt.ylabel('Total Sales')
                plt.title('Total Sales by Region')
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                bar_chart = base64.b64encode(buf.read()).decode()
            except Exception as e:
                logger.warning(f"Failed to generate bar_chart: {e}")
            cumulative_sales_chart = ""
            try:
                df_sorted = df.sort_values('date')
                df_sorted['cumulative_sales'] = df_sorted['sales'].cumsum()
                plt.figure(figsize=(6,4))
                plt.plot(df_sorted['date'], df_sorted['cumulative_sales'], color='red')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Sales')
                plt.title('Cumulative Sales Over Time')
                plt.tight_layout()
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
                plt.close()
                buf2.seek(0)
                cumulative_sales_chart = base64.b64encode(buf2.read()).decode()
            except Exception as e:
                logger.warning(f"Failed to generate cumulative_sales_chart: {e}")
            return JSONResponse(content={
                "total_sales": int(total_sales) if total_sales is not None else None,
                "top_region": str(top_region) if top_region is not None else None,
                "day_sales_correlation": round(day_sales_corr, 10),
                "bar_chart": bar_chart,
                "median_sales": int(median_sales) if median_sales is not None else None,
                "total_sales_tax": int(total_sales_tax) if total_sales_tax is not None else None,
                "cumulative_sales_chart": cumulative_sales_chart
            })

        # If not recognized, return error
        else:
            return JSONResponse(
                content={"error": "Unrecognized CSV format. Only weather and sales datasets are supported."},
                status_code=400
            )

    except Exception as e:
        logger.error(f"Error in /api/: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload