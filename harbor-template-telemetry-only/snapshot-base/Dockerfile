FROM python:3.13-slim

WORKDIR /app

RUN mkdir -p /logs/verifier

# Install Python dependencies used by the verifier (tests/check_prediction.py)
RUN pip install --no-cache-dir openai tabulate

# Install Docker CLI (needed for docker compose inside the container)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
      > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# Bake in the OpenTelemetry Demo source and ALL snapshot data so that
# any Harbor task can select the right Prometheus snapshot at runtime
# via the SNAPSHOT_NAME env var.
COPY opentelemetry-demo/ /app/opentelemetry-demo/
COPY data/ /app/data/
COPY docker-compose-base.yml /app/docker-compose-base.yml
COPY docker-compose.snapshot.yml /app/docker-compose.snapshot.yml
COPY otelcol-config-snapshot.yml /app/otelcol-config-snapshot.yml
COPY jaeger-config-snapshot.yml /app/jaeger-config-snapshot.yml

COPY entrypoint.sh /app/entrypoint.sh
COPY check_health.sh /app/check_health.sh
RUN chmod +x /app/entrypoint.sh /app/check_health.sh

# Non-root user for the agent. No docker group → no /var/run/docker.sock access.
RUN useradd --create-home --shell /bin/bash --uid 1001 agent \
 && mkdir -p /agent /home/agent/.cache \
 && chown -R agent:agent /agent /home/agent

# Block the agent from reading the baked-in otel-demo source.
# /app/opentelemetry-demo is only read once by the root entrypoint (cp -a).
RUN chown root:root /app/opentelemetry-demo && chmod 700 /app/opentelemetry-demo

# Block shells until the entrypoint has finished bringing up all services.
# BASH_ENV is sourced by non-interactive `bash -c` (used by Harbor's exec);
# /etc/profile.d/ covers interactive login shells (manual debugging).
RUN printf '#!/bin/bash\n\
# Skip the wait gate inside the entrypoint itself (PID 1) to avoid a\n\
# deadlock: the entrypoint is the process that *creates* /tmp/env-ready.\n\
if [ "$$" != "1" ] && [ ! -f /tmp/env-ready ]; then\n\
  echo "[wait-env] Waiting for environment to be ready..." >&2\n\
  while [ ! -f /tmp/env-ready ]; do sleep 2; done\n\
  echo "[wait-env] Environment is ready." >&2\n\
fi\n\
# Source dynamic port mappings assigned by the entrypoint\n\
if [ -f /tmp/env-ports ]; then\n\
  . /tmp/env-ports\n\
fi\n' > /etc/profile.d/wait-env-ready.sh && chmod +x /etc/profile.d/wait-env-ready.sh
ENV BASH_ENV=/etc/profile.d/wait-env-ready.sh

ENTRYPOINT ["/app/entrypoint.sh"]
