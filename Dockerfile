FROM python:3.12-slim

WORKDIR /app

# Install system deps for pandas/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e ".[data-yfinance]"

# Copy strategies, examples, and config
COPY strategies/ strategies/
COPY examples/ examples/

# Default command: run demo
ENTRYPOINT ["alphaevo"]
CMD ["demo"]
