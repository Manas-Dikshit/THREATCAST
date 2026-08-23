FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend + the layers it serves (data pipeline, world model)
COPY backend/ ./backend/
COPY data_pipeline/ ./data_pipeline/
COPY ml/__init__.py ml/src ./ml/
COPY ml/configs ./ml/configs
COPY ml/artifacts ./ml/artifacts

EXPOSE 8000
WORKDIR /app/backend
# Apply DB migrations, then serve.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
