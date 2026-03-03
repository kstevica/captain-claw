FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY botport/ botport/

RUN pip install --no-cache-dir .

RUN mkdir -p /data

VOLUME ["/data"]

EXPOSE 33080

CMD ["botport"]
