FROM python:3.13-slim

RUN pip install --no-cache-dir 'mlflow==3.16.0'

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////mlflow.db \
    --artifacts-destination /mlartifacts \
    --allowed-hosts "*"