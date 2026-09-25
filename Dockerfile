# Ubuntu image with everything needed to run the demo scripts.
#
#   docker compose build app
#   docker compose run --rm app python -m scripts.setup_db
#   docker compose run --rm app python run_demo.py s1
#   docker compose run --rm app python -m pytest tests
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip libpq5 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 marks the system Python as externally managed, so use a venv.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run_demo.py", "algorithm1"]
