FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf-cache

# CPU-only PyTorch keeps the image small (no GPU needed at this catalog size).
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Download the model once at build time so the container starts offline.
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-small')"

COPY core.py index.py app.py evaluate.py ./
COPY static ./static

# `catalog/` and `data/` are mounted as volumes (see docker-compose.yml).
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
