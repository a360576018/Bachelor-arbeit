FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY evaluate.py /app/evaluate.py

RUN python -c "import torch, pandas, anomalib, requests; \
print('PyTorch:', torch.__version__); \
print('CUDA build:', torch.version.cuda); \
print('Pandas:', pandas.__version__); \
print('Anomalib:', anomalib.__version__)"

ENTRYPOINT ["python", "/app/evaluate.py"]