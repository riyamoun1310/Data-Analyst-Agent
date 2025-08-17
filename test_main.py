import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Data Analyst Agent API is running" in resp.json()["message"]

def test_weather_csv():
    with open("test_sample_weather.csv", "rb") as f:
        resp = client.post("/api/", files={"file": f})
        assert resp.status_code == 200
        data = resp.json()
        assert "average_temp_c" in data
        assert "temp_line_chart" in data

def test_sales_csv():
    with open("test_sample_sales.csv", "rb") as f:
        resp = client.post("/api/", files={"file": f})
        assert resp.status_code == 200
        data = resp.json()
        assert "total_sales" in data
        assert "bar_chart" in data

def test_invalid_csv():
    import io
    bad_csv = io.BytesIO(b"foo,bar\n1,2\n3,4")
    resp = client.post("/api/", files={"file": ("bad.csv", bad_csv)})
    assert resp.status_code == 400
    assert "error" in resp.json()
