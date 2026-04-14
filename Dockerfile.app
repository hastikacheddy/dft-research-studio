# Dockerfile.app
# ---------------
# Production image for the Gradio frontend.
#
# Build:
#   docker build -f Dockerfile.app -t dft-app .
#
# Run (expects LitServe to be reachable at LITSERVE_URL):
#   docker run -p 7860:7860 \
#     -e LITSERVE_URL=http://dft-serve:8000/predict \
#     dft-app

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip install litserve>=0.2.0

# The app only needs the package + app.py (engines live in the serve container)
COPY dft_research_studio/ /app/dft_research_studio/
COPY app.py               /app/app.py

ENV LITSERVE_URL=http://dft-serve:8000/predict \
    GRADIO_PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" \
    || exit 1

# No --standalone: delegates all inference to LitServe
CMD ["python", "app.py", "--host", "0.0.0.0"]
