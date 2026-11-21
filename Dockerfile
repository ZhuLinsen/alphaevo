FROM python:3.12-slim

WORKDIR /app

# Install build deps needed by numpy/pandas extensions on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy full source & bundled assets before non-editable install
COPY pyproject.toml README.md SKILL.md ./
COPY src/ src/
COPY strategies/ strategies/
COPY examples/ examples/

RUN pip install --no-cache-dir ".[data-yfinance]"

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3)" || exit 1

ENTRYPOINT ["alphaevo"]
CMD ["web", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
