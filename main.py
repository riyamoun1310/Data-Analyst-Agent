from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
import uvicorn
import io
import json
import re
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from bs4 import BeautifulSoup
import duckdb

app = FastAPI()

# Helper: Scrape Wikipedia table
def scrape_wikipedia_table(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    table = soup.find('table', {'class': 'wikitable'})
    df = pd.read_html(str(table))[0]
    return df

# Helper: Plot and encode as base64
def plot_scatter_with_regression(x, y, xlabel, ylabel):
    plt.figure(figsize=(6,4))
    plt.scatter(x, y)
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, 'r--', label='Regression line')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=120)
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()
    data_uri = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode()
    return data_uri

# Helper: Query DuckDB on remote parquet
def duckdb_query_count_cases():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    query = """
    SELECT court, COUNT(*) as n_cases FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
    WHERE year BETWEEN 2019 AND 2022
    GROUP BY court ORDER BY n_cases DESC
    """
    df = con.execute(query).fetchdf()
    top_court = df.iloc[0]['court']
    return top_court

def duckdb_query_regression_and_plot():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    query = """
    SELECT year, date_of_registration, decision_date FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=33_10/bench=*/metadata.parquet?s3_region=ap-south-1')
    """
    df = con.execute(query).fetchdf()
    # Calculate delay in days
    df['date_of_registration'] = pd.to_datetime(df['date_of_registration'], errors='coerce')
    df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')
    df['delay'] = (df['decision_date'] - df['date_of_registration']).dt.days
    # Regression slope by year
    grouped = df.groupby('year')['delay'].mean().reset_index()
    x = grouped['year']
    y = grouped['delay']
    if len(x) > 1:
        m, b = np.polyfit(x, y, 1)
        slope = float(m)
    else:
        slope = None
    # Plot
    img = plot_scatter_with_regression(x, y, 'Year', 'Avg Delay (days)')
    return slope, img

@app.post("/api/")
async def analyze(request: Request, file: UploadFile = File(None)):
    if file:
        task = (await file.read()).decode()
    else:
        task = (await request.body()).decode()
    # --- Simple parser for Wikipedia film question ---
    if 'wikipedia.org/wiki/List_of_highest-grossing_films' in task:
        url = 'https://en.wikipedia.org/wiki/List_of_highest-grossing_films'
        df = scrape_wikipedia_table(url)
        # Clean up columns
        df.columns = [c.lower().replace('\u200b', '').strip() for c in df.columns]
        # 1. How many $2 bn movies were released before 2020?
        df['worldwide'] = df['worldwide'].replace('[\$,]', '', regex=True).astype(float)
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        count_2bn = int(((df['worldwide'] >= 2_000_000_000) & (df['year'] < 2020)).sum())
        # 2. Earliest film over $1.5bn
        over_1_5 = df[df['worldwide'] > 1_500_000_000]
        earliest = over_1_5.sort_values('year').iloc[0]['title']
        # 3. Correlation between Rank and Peak
        if 'peak' in df.columns:
            corr = float(np.corrcoef(df['rank'], df['peak'])[0,1])
        else:
            corr = None
        # 4. Scatterplot
        if 'peak' in df.columns:
            img = plot_scatter_with_regression(df['rank'], df['peak'], 'Rank', 'Peak')
        else:
            img = ''
        return JSONResponse([count_2bn, earliest, corr, img])
    # --- Indian High Court Judgments Example ---
    if 'indian high court judgement dataset' in task.lower():
        # 1. Which high court disposed the most cases from 2019 - 2022?
        top_court = duckdb_query_count_cases()
        # 2. Regression slope of delay by year in court=33_10
        slope, img = duckdb_query_regression_and_plot()
        # 3. Return JSON object as required
        result = {
            "Which high court disposed the most cases from 2019 - 2022?": top_court,
            "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": slope,
            "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": img
        }
        return JSONResponse(result)
    # --- Add more task handlers here ---
    return JSONResponse(["Not implemented", "", 0.0, ""])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
