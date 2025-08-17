
# Sales analysis service
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from fastapi.responses import JSONResponse

def analyze_sales(df: pd.DataFrame):
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
    except Exception:
        pass
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
    except Exception:
        bar_chart = ""
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
    except Exception:
        cumulative_sales_chart = ""
    return {
        "total_sales": int(total_sales) if total_sales is not None else None,
        "top_region": str(top_region) if top_region is not None else None,
        "day_sales_correlation": round(day_sales_corr, 10),
        "bar_chart": bar_chart,
        "median_sales": int(median_sales) if median_sales is not None else None,
        "total_sales_tax": int(total_sales_tax) if total_sales_tax is not None else None,
        "cumulative_sales_chart": cumulative_sales_chart
    }
