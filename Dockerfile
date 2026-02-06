# GRU Stock Predictor API - Docker deploy for render
# Keeps .git file in the image so push to git can happen after finetuning/retraining(git remote origin)
FROM python:3.12-slim

WORKDIR /app

# Copy entire repo (including .git)
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

#Render sets PORT at runtime (default 10000)
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]