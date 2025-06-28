# Data Analyst Agent API

This API uses LLMs and Python data tools to source, prepare, analyze, and visualize data from arbitrary sources, including Wikipedia and remote datasets.

## Features
- Accepts a POST request with a data analysis task description (plain text or file)
- Scrapes, analyzes, and visualizes data as requested
- Returns answers in the requested format (JSON, base64-encoded images, etc.)

## Example Usage
```
curl -X POST "https://<your-app-url>/api/" -F "@question.txt"
```

## Deployment
- Python 3.9+
- Install dependencies: `pip install -r requirements.txt`
- Run locally: `uvicorn main:app --host 0.0.0.0 --port 8000`

## License
MIT
