FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download BGE model at BUILD time so server starts fast
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('BAAI/bge-large-en-v1.5', cache_folder='/app/model_cache'); \
print('BGE model ready!')"

COPY backend/ ./backend/

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]