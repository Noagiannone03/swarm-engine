FROM rust:1.91.0-slim-bookworm AS native-builder

ARG PARALLAX_REPO_URL=https://github.com/Noagiannone03/swarm-engine.git
ARG PARALLAX_COMMIT

RUN test -n "${PARALLAX_COMMIT}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/build-venv \
    && /opt/build-venv/bin/pip install --no-cache-dir maturin==1.14.1

RUN git clone --filter=blob:none --no-checkout "${PARALLAX_REPO_URL}" /opt/parallax \
    && git -C /opt/parallax fetch --depth 1 origin "${PARALLAX_COMMIT}" \
    && git -C /opt/parallax checkout --detach FETCH_HEAD \
    && git -C /opt/parallax rev-parse HEAD | grep -qx "${PARALLAX_COMMIT}" \
    && cd /opt/parallax/native/fabi-network \
    && /opt/build-venv/bin/maturin build \
        --release \
        --interpreter /opt/build-venv/bin/python \
        --out /opt/wheels

FROM python:3.12-slim-bookworm

ARG PARALLAX_COMMIT

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PARALLAX_SRC=/opt/parallax

RUN test -n "${PARALLAX_COMMIT}" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Keep the heavyweight scheduler runtime separate from source changes.
RUN pip install --upgrade pip setuptools wheel \
    && pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install "mlx[cpu]==0.31.2" "mlx-lm==0.31.3"

COPY --from=native-builder /opt/parallax /opt/parallax
COPY --from=native-builder /opt/wheels /opt/fabi-network-wheels

RUN pip install /opt/fabi-network-wheels/fabi_network_native-*.whl \
    && pip install /opt/parallax \
    && python -c "import fabi_network_native; assert fabi_network_native.PROTOCOL_VERSION == 1"

COPY deploy/scheduler/entrypoint.sh /usr/local/bin/parallax-scheduler-entrypoint
RUN chmod +x /usr/local/bin/parallax-scheduler-entrypoint

WORKDIR /opt/parallax-runtime

EXPOSE 3001/tcp 18080/tcp 18080/udp

LABEL org.opencontainers.image.source="https://github.com/Noagiannone03/swarm-engine" \
      org.opencontainers.image.revision="${PARALLAX_COMMIT}"

ENTRYPOINT ["parallax-scheduler-entrypoint"]
