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

# MKL runtime — provides libmkl_def.so.2 and other computational kernels
# loaded via dlopen (not bundled by auditwheel)
RUN uv pip install --system --no-cache mkl

# ILP64 numpy from fork index (must go after MKL, before jamma)
RUN uv pip install --system --no-cache numpy \
    --extra-index-url https://michael-denyer.github.io/numpy-mkl \
    --reinstall

# JAMMA runtime deps (everything except numpy, which is ILP64 above)
RUN uv pip install --system --no-cache \
    psutil loguru threadpoolctl jax jaxlib jaxtyping click progressbar2 bed-reader

# JAMMA itself — --no-deps to preserve ILP64 numpy
RUN uv pip install --system --no-cache --no-deps jamma

# Verify ILP64 at build time
RUN python -c "\
import numpy as np; \
cfg = np.show_config(mode='dicts'); \
blas = cfg['Build Dependencies']['blas']['name']; \
assert 'ilp64' in blas.lower(), f'Expected ILP64 BLAS, got: {blas}'; \
print(f'ILP64 verified: {blas}')"

ENTRYPOINT ["jamma"]
CMD ["--help"]
