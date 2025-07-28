# Data Analyst Agent API

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd Data-Analyst-Agent

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Run the application
python main.py
```

## 📋 API Documentation

### Endpoints

#### `GET /`
Health check endpoint
- **Response**: `{"message": "Data Analyst Agent API is running", "status": "healthy"}`

#### `GET /health`
Detailed health check with database status
- **Response**: System health information

#### `POST /api/`
Main analysis endpoint
- **Input**: Text task description (via request body or file upload)
- **Content-Type**: `text/plain` or `multipart/form-data`
- **Response**: JSON with analysis results

### Supported Analysis Tasks

#### 1. Wikipedia Films Analysis
**Trigger**: Include `wikipedia.org/wiki/list_of_highest-grossing_films` in your request

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/" \
  -H "Content-Type: text/plain" \
  -d "Analyze wikipedia.org/wiki/list_of_highest-grossing_films data"
```

**Response Format**:
```json
{
  "movies_2bn_before_2020": 5,
  "earliest_1_5bn_film": "Avatar",
  "rank_peak_correlation": -0.85,
  "scatterplot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "data_points_analyzed": 50
}
```

#### 2. Indian High Court Analysis
**Trigger**: Include `indian high court` or `high court` in your request

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/" \
  -H "Content-Type: text/plain" \
  -d "Analyze indian high court judgement dataset"
```

**Response Format**:
```json
{
  "top_court_2019_2022": "Delhi High Court",
  "regression_slope_court_33_10": 2.5,
  "delay_trend_plot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## 🏗️ Architecture

### Technologies Used
- **FastAPI**: Modern web framework for APIs
- **DuckDB**: In-process analytical database
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **BeautifulSoup**: Web scraping
- **Uvicorn**: ASGI server

### Key Features
- ✅ **Async/await support** for better performance
- ✅ **Comprehensive error handling** with proper HTTP status codes
- ✅ **Input validation** and sanitization
- ✅ **Database connection pooling**
- ✅ **Configurable timeouts** and size limits
- ✅ **CORS support** for web applications
- ✅ **Logging** for debugging and monitoring
- ✅ **Health checks** for deployment monitoring

## 🔧 Configuration

### Environment Variables
See `.env.example` for all available configuration options:

- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: INFO)
- `MAX_FILE_SIZE`: Maximum upload file size in bytes
- `REQUEST_TIMEOUT`: HTTP request timeout in seconds

### Security Considerations
- File size limits prevent DoS attacks
- Request timeouts prevent hanging connections
- Input validation prevents injection attacks
- CORS configuration should be restricted in production

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production Deployment

#### Using Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Using Heroku
The `Procfile` is already configured for Heroku deployment:
```bash
git push heroku main
```

## 🧪 Testing

### Manual Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test Wikipedia analysis
curl -X POST "http://localhost:8000/api/" \
  -H "Content-Type: text/plain" \
  -d "Analyze wikipedia.org/wiki/list_of_highest-grossing_films"

# Test file upload
echo "Analyze indian high court data" > task.txt
curl -X POST "http://localhost:8000/api/" \
  -F "file=@task.txt"
```

## 📊 Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `413`: Payload Too Large
- `500`: Internal Server Error
- `501`: Not Implemented
- `502`: Bad Gateway (external service error)

Error responses include detailed messages:
```json
{
  "detail": "File too large. Maximum size: 10485760 bytes"
}
```

## 🔍 Monitoring

### Logs
The application logs important events:
- Request processing
- Database queries
- Errors and warnings
- Performance metrics

### Health Checks
- `/health` endpoint provides system status
- Database connectivity verification
- Timestamp for monitoring freshness

## 📄 License
MIT
