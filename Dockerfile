FROM python:3.11-slim
WORKDIR /app
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ads ./ads
CMD ["uvicorn", "ads.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
