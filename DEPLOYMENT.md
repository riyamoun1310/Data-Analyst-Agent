# 🚀 Deployment Guide

## Quick Deployment Options

### 1. 🐳 **Docker Deployment (Recommended)**

#### Local Docker Build & Run:
```bash
# Build the image
docker build -t data-analyst-agent .

# Run the container
docker run -p 8000:8000 data-analyst-agent
```

#### Using Docker Compose:
```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### 2. ☁️ **Cloud Platform Deployments**

#### **Heroku** (Easiest):
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-data-analyst-agent

# Set environment variables
heroku config:set LOG_LEVEL=INFO
heroku config:set MAX_FILE_SIZE=10485760

# Deploy
git push heroku main

# Open your app
heroku open
```

#### **Railway**:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

#### **Google Cloud Run**:
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/data-analyst-agent

# Deploy to Cloud Run
gcloud run deploy data-analyst-agent \
  --image gcr.io/YOUR_PROJECT_ID/data-analyst-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

#### **AWS App Runner**:
```bash
# Push to ECR first
aws ecr create-repository --repository-name data-analyst-agent
docker tag data-analyst-agent:latest YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/data-analyst-agent:latest
docker push YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/data-analyst-agent:latest

# Then create App Runner service via AWS Console
```

### 3. 🖥️ **VPS/Server Deployment**

#### Using systemd (Linux):
```bash
# Copy files to server
scp -r . user@yourserver:/opt/data-analyst-agent/

# Install dependencies
cd /opt/data-analyst-agent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create systemd service
sudo nano /etc/systemd/system/data-analyst-agent.service
```

Service file content:
```ini
[Unit]
Description=Data Analyst Agent API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/data-analyst-agent
Environment=PATH=/opt/data-analyst-agent/venv/bin
ExecStart=/opt/data-analyst-agent/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Start service
sudo systemctl daemon-reload
sudo systemctl enable data-analyst-agent
sudo systemctl start data-analyst-agent
```

### 4. 🌐 **Reverse Proxy Setup (Nginx)**

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 📊 **Monitoring & Health Checks**

### Health Check Endpoints:
- `GET /` - Basic health check
- `GET /health` - Detailed health with database status

### Monitoring Commands:
```bash
# Check if service is running
curl http://your-domain.com/health

# Monitor logs in Docker
docker logs -f container_name

# Monitor system resources
htop
```

## 🔒 **Production Security Checklist**

- [ ] Set proper CORS origins (not `*`)
- [ ] Use HTTPS with SSL certificates
- [ ] Set up rate limiting
- [ ] Configure proper logging
- [ ] Set environment variables securely
- [ ] Use secrets management
- [ ] Set up monitoring/alerts
- [ ] Regular security updates

## 🔧 **Environment Variables**

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your production values
```

## 📈 **Performance Optimization**

- Use gunicorn for production: `gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker`
- Set up Redis for caching
- Use CDN for static assets
- Configure database connection pooling
- Set up load balancing for multiple instances

## 🚨 **Troubleshooting**

### Common Issues:

1. **Port already in use**: `lsof -i :8000`
2. **Permission denied**: Check file permissions
3. **Memory issues**: Increase container/server memory
4. **Database connection**: Check network connectivity

### Debug Commands:
```bash
# Check application logs
docker logs data-analyst-agent

# Interactive shell in container
docker exec -it data-analyst-agent /bin/bash

# Test API endpoints
curl -X POST http://localhost:8000/api/ -d "test task"
```
