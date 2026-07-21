# Slim JAMMA image with ILP64 numpy (MKL) for large-scale GWAS
#
# MKL is x86_64-only — always build with --platform linux/amd64:
#   docker build --platform linux/amd64 -t jamma .
#
# Run:
#   docker run --platform linux/amd64 -v $(pwd)/data:/data jamma -gk 1 -bfile /data/study -o /data/output
#   docker run --platform linux/amd64 -v $(pwd)/data:/data jamma -lmm 1 -bfile /data/study -k /data/k.cXX.txt -o /data/output
FROM python:3.11-slim

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install ILP64 numpy from custom index, then runtime deps and jamma:
# 1. mkl runtime libraries
# 2. ILP64 numpy from michael-denyer/numpy-mkl (must not be overwritten).
#    Capped <2.5: the index also serves 2.5.x, and numba (via shap) in the
#    downstream Databricks stack requires numpy<2.5. On this py3.11 base pip
#    already falls back to 2.4.x (2.5.x needs >=3.12), but the explicit cap
#    keeps that guarantee if the base image ever moves to Python 3.12+.
# 3. Runtime deps (--no-deps on jamma to prevent numpy downgrade back to LP64)
RUN uv pip install --system --no-cache mkl && \
    uv pip install --system --no-cache 'numpy<2.5' \
        --index-url https://michael-denyer.github.io/numpy-mkl \
        --reinstall && \
    uv pip install --system --no-cache \
        psutil loguru threadpoolctl click progressbar2 bed-reader && \
    uv pip install --system --no-cache --no-deps jamma

# Verify ILP64 at build time
RUN python -c "\
import numpy as np; \
cfg = np.show_config(mode='dicts'); \
blas = cfg['Build Dependencies']['blas']['name']; \
assert 'ilp64' in blas.lower(), f'Expected ILP64 BLAS, got: {blas}'; \
print(f'ILP64 verified: {blas}')"

# Run as non-root for security (volume mounts won't be owned by root)
RUN useradd -m -u 1000 jamma && mkdir -p /data && chown jamma:jamma /data
USER jamma

ENTRYPOINT ["jamma"]
CMD ["--help"]
