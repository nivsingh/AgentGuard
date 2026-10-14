FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so this layer is cached until requirements.txt changes.
# Java is for the Spark warehouse layer (python -m warehouse). The app and the agent do not need it.
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-warehouse.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY . .

# Bake the synthetic dataset into the image (the generator is seeded, so it is reproducible).
RUN python -m data.generate

ENV AGENTGUARD_STATE=/app/state
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# The demo app. The test suite runs with: docker compose run --rm test
CMD ["streamlit", "run", "app/streamlit_app.py"]
