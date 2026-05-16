FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY ackknow_a1 /app/ackknow_a1
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "ackknow_a1.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
