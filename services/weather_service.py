
# Weather analysis service
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from fastapi.responses import JSONResponse

def analyze_weather(df: pd.DataFrame):
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
    except Exception:
        temp_line_chart = ""
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
    except Exception:
        precip_histogram = ""
    return {
        "average_temp_c": round(avg_temp, 2) if avg_temp is not None else None,
        "max_precip_date": max_precip_date,
        "min_temp_c": min_temp,
        "temp_precip_correlation": round(temp_precip_corr, 10),
        "average_precip_mm": round(avg_precip, 2) if avg_precip is not None else None,
        "temp_line_chart": temp_line_chart,
        "precip_histogram": precip_histogram
    }
