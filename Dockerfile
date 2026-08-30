# Reproducible linux/amd64 JAMMA image with MKL-backed ILP64 NumPy.
# The build-capable stage compiles this checkout's native extensions; the
# runtime stage remains slim and receives only /usr/local from the builder.
FROM python:3.11.13-bookworm@sha256:e75be128195ec5b78912d55646e87d4638fc95234302a34472aeb2a474334cb1 AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY docker/requirements-container.txt /tmp/requirements-container.txt
RUN python -m pip install --no-cache-dir --no-deps \
        -r /tmp/requirements-container.txt
RUN python -m pip install --no-cache-dir --no-deps \
        numpy==2.4.6 \
        mkl-service==2.7.2 \
        --index-url https://michael-denyer.github.io/numpy-mkl \
        --force-reinstall

# Copy only the inputs needed to build the checked-out project. Installing the
# published `jamma` distribution here would decouple the image from this source.
COPY pyproject.toml README.md hatch_build.py ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps .

RUN python -c "\
import numpy as np; \
cfg = np.show_config(mode='dicts'); \
blas = cfg['Build Dependencies']['blas']['name']; \
assert 'ilp64' in blas.lower(), f'Expected ILP64 BLAS, got: {blas}'; \
print(f'ILP64 verified: {blas}')"

FROM python:3.11.13-slim-bookworm@sha256:cec9aa7aa96eea4fa036e9b82be1e6b325f2e3707f462d885868df51ec0a4b47

COPY --from=build /usr/local /usr/local

RUN useradd -m -u 1000 jamma && mkdir -p /data && chown jamma:jamma /data
USER jamma

ENTRYPOINT ["jamma"]
CMD ["--help"]
