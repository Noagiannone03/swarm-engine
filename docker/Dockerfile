ARG PARALLAX_BASE_IMAGE=lmsysorg/sglang:v0.5.7
FROM ${PARALLAX_BASE_IMAGE}

ENV SGLANG_ENABLE_JIT_DEEPGEMM=0

WORKDIR /parallax

# A qualified Parallax image may be supplied as the build base for a small
# source-only lab rollout. Remove its editable source tree first so deleted
# modules cannot survive in the resulting image.
RUN rm -rf /parallax/src

COPY README.md ./README.md
COPY src ./src
COPY pyproject.toml ./pyproject.toml

RUN pip install -e '.[gpu]'
