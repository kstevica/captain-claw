FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY captain_claw/ captain_claw/
COPY skills/ skills/

RUN pip install --no-cache-dir .

RUN mkdir -p /data/workspace /data/sessions /data/skills

VOLUME ["/data/workspace", "/data/sessions", "/data/skills"]

COPY docker-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["docker-entrypoint.sh"]

EXPOSE 23080

CMD ["captain-claw-web"]
