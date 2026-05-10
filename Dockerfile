# =====================================================
# Dockerfile  --  builds an image that runs the FastAPI app
# =====================================================

# 1. Python base image (slim = smaller image size)
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy and install dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the project
COPY . .

# 5. Expose FastAPI's port
EXPOSE 8000

# 6. Run FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
