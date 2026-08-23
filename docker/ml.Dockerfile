# ML training/inference image (CPU base; CUDA base chosen later if GPU-in-container is required).
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./ml/
COPY data_pipeline/ ./data_pipeline/
ENV PYTHONPATH=/app/ml/src:/app/data_pipeline/src

CMD ["python", "-c", "print('THREATCAST ML image - training entrypoint arrives in a later phase')"]
