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

# MKL + deps + ILP64 numpy + JAMMA in minimal layers
# Order matters: MKL first, then ILP64 numpy (--reinstall), then deps, then jamma (--no-deps)
RUN uv pip install --system --no-cache mkl && \
    uv pip install --system --no-cache numpy \
        --extra-index-url https://michael-denyer.github.io/numpy-mkl \
        --reinstall && \
    uv pip install --system --no-cache \
        psutil loguru threadpoolctl jax jaxlib jaxtyping click progressbar2 bed-reader && \
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
