# 🎯 Data Analyst Agent API

**Author:** Riya Moun  
**Project:** Advanced Data Analytics System  
**Institution:** Academic Project Demonstration

---

## 🚀 Overview

The **Data Analyst Agent API** is a professional-grade FastAPI backend for automated data analysis and visualization. It is designed to process weather and sales CSV datasets, generate real-time statistical insights, and deliver high-quality visualizations—all through a simple API.

---

## 🌟 Key Features

- **Automated CSV Analysis:** Upload a CSV and get instant, real-coded analysis (no hardcoding).
- **Dataset Auto-Detection:** Supports both weather and sales datasets, detected by column names.
- **Statistical Insights:** Computes means, medians, correlations, and more from real data.
- **Dynamic Visualizations:** Generates base64-encoded matplotlib charts on the fly.
- **Robust Error Handling:** Graceful responses for invalid input or unrecognized formats.
- **Production-Ready:** Includes health checks, CORS, logging, and Docker deployment support.

---

## 🛠️ API Endpoints

### Health Check

- **GET /** or **GET /health**
- Returns API status and version.

### Main Analysis Endpoint

- **POST /api/**
- Accepts: CSV file upload (`form-data`, key: `file`)
- Returns: JSON with all required keys for weather or sales datasets.

#### Example (using `curl`):

```bash
curl -X POST "http://localhost:8000/api/" -F "file=@yourfile.csv"
```

---

## 📊 Example Output

**Weather CSV:**
```json
{
  "average_temp_c": 23.5,
  "max_precip_date": "2023-07-15",
  "min_temp_c": 17.2,
  "temp_precip_correlation": 0.42,
  "average_precip_mm": 12.7,
  "temp_line_chart": "<base64-image>",
  "precip_histogram": "<base64-image>"
}
```

**Sales CSV:**
```json
{
  "total_sales": 150000,
  "top_region": "North",
  "day_sales_correlation": 0.12,
  "bar_chart": "<base64-image>",
  "median_sales": 1200,
  "total_sales_tax": 15000,
  "cumulative_sales_chart": "<base64-image>"
}
```

---

## 🐳 Deployment & Local Run

- **Local Development:**
  ```bash
  pip install -r requirements.txt
  uvicorn main:app --reload
  ```
- **Production:**
  ```bash
  python main.py
  ```
- **Docker:**
  ```bash
  docker build -t data-analyst-agent .
  docker run -p 8000:8000 data-analyst-agent
  ```

---

## 🧪 Testing

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Manual Test:** Use `curl` or Postman to POST a CSV file to `/api/`.

---

## 🔒 Security & Best Practices

- CORS is open for development; restrict in production.
- All outputs are computed from real uploaded files—no hardcoded values.
- Handles large files and invalid formats gracefully.

---

## 👩‍💻 About the Developer

**Riya Moun**  
- Advanced Data Science & Full-Stack Development  
- Specializations: Python, FastAPI, Statistical Analysis, Visualization  
- Project Philosophy: Real-world, production-ready, and robust solutions

---

## 📞 Contact

- **GitHub:** [riyamoun1310](https://github.com/riyamoun1310/Data-Analyst-Agent)
- **Live Demo:** [https://data-analyst-agent-ee6j.onrender.com](https://data-analyst-agent-ee6j.onrender.com)

---

*Built with passion for data science, software