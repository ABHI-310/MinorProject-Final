FROM node:22-bookworm-slim AS dashboard
WORKDIR /frontend
COPY vendor/real_minor_frontend/ ./
COPY frontend/ ./
RUN npm ci && VITE_API_URL=/ npm run build

FROM python:3.11-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY configs/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r /tmp/requirements.txt
COPY src/ src/
COPY pcb_rul/config.py pcb_rul/features.py pcb_rul/ensemble.py pcb_rul/train_gru.py pcb_rul/
COPY examples/ examples/
COPY --from=dashboard /frontend/dist/ frontend/dist/
COPY artifacts/models/pcb_defectdet_k100_r100_missing_hole_layout11/ artifacts/models/pcb_defectdet_k100_r100_missing_hole_layout11/
COPY artifacts/models/pcb_rul_ensemble/ artifacts/models/pcb_rul_ensemble/
COPY data/synthetic/phase3_rul/scenarios.json data/synthetic/phase3_rul/scenarios.json
ENV PYTHONPATH=/app/src PCB_FRONTEND_BUILD=/app/frontend/dist PCB_DEVICE=cpu YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
