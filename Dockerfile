# Slim JAMMA image with ILP64 numpy (MKL) for large-scale GWAS
#
# Build:  docker build -t jamma .
# Run:    docker run -v $(pwd)/data:/data jamma jamma -o /data/output lmm -bfile /data/study -k /data/k.cXX.txt -lmm 1
FROM python:3.11-slim

# MKL runtime — provides libmkl_def.so.2 and other computational kernels
# loaded via dlopen (not bundled by auditwheel)
RUN pip install --no-cache-dir mkl

# ILP64 numpy from fork index (must go after MKL, before jamma)
RUN pip install --no-cache-dir numpy \
    --extra-index-url https://michael-denyer.github.io/numpy-mkl \
    --force-reinstall --upgrade

# JAMMA runtime deps (everything except numpy, which is ILP64 above)
RUN pip install --no-cache-dir \
    psutil loguru threadpoolctl jax jaxlib jaxtyping typer progressbar2 bed-reader

# JAMMA itself — --no-deps to preserve ILP64 numpy
RUN pip install --no-cache-dir --no-deps jamma

# Verify ILP64 at build time
RUN python -c "\
import numpy as np; \
cfg = np.show_config(mode='dicts'); \
blas = cfg['Build Dependencies']['blas']['name']; \
assert 'ilp64' in blas.lower(), f'Expected ILP64 BLAS, got: {blas}'; \
print(f'ILP64 verified: {blas}')"

ENTRYPOINT ["python", "-m"]
CMD ["jamma", "--help"]
