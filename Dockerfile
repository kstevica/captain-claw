FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY captain_claw/ captain_claw/
COPY skills/ skills/

RUN pip install --no-cache-dir .

RUN playwright install --with-deps chromium

RUN pip install --no-cache-dir pocket-tts

# Create non-root user
RUN groupadd -r claw && useradd -r -g claw -m -d /home/claw claw

RUN mkdir -p /data/workspace /data/sessions /data/skills /home/claw/.cache \
    && chown -R claw:claw /data /app /home/claw

VOLUME ["/data/workspace", "/data/sessions", "/data/skills"]

COPY docker-entrypoint.sh /usr/local/bin/

# Switch to non-root user
USER claw

ENTRYPOINT ["docker-entrypoint.sh"]

EXPOSE 23080

CMD ["captain-claw-web"]
