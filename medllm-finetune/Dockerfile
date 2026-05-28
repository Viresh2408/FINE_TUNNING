FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API files
COPY api/ /app/api/

# Expose ports for both FastAPI and Gradio
EXPOSE 8000
EXPOSE 7860

# We use a script or docker-compose to run either/both, but by default run the API
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
