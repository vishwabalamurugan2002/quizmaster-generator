FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY daily_generator.py .
RUN mkdir -p backups

CMD ["python", "daily_generator.py"]