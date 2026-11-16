FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends iperf3 make && rm -rf /var/lib/apt/lists/*
WORKDIR /lab
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "grpcio>=1.60" "grpcio-tools>=1.60" "prometheus-client>=0.19"
COPY proto proto
COPY failover_lab failover_lab
COPY Makefile .
RUN make proto
ENV PYTHONUNBUFFERED=1
