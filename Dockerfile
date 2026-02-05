# Databricks Container Services image for JAMMA benchmarks
# Bakes ILP64 numpy + jamma so notebooks don't need pip installs or kernel restarts.
#
# Build: docker build -t jamma-dbr .
# DBR 16.4 LTS: docker build --build-arg DBR_VERSION=16.4-LTS -t jamma-dbr .
ARG DBR_VERSION=15.4-LTS
FROM databricksruntime/standard:${DBR_VERSION}

# Purge OpenBLAS — segfaults on matrices >50k and conflicts with MKL
RUN apt-get purge -y libopenblas* libblas* 2>/dev/null || true

# MKL runtime — provides libmkl_def.so.2 and other computational kernels
# that auditwheel can't bundle (loaded via dlopen, not ELF NEEDED)
RUN /databricks/python3/bin/pip install --no-cache-dir mkl

# ILP64 numpy from fork index — must go after MKL, before anything that pulls numpy from PyPI
RUN /databricks/python3/bin/pip install --no-cache-dir \
    numpy --extra-index-url https://michael-denyer.github.io/numpy-mkl \
    --force-reinstall --upgrade

# Jamma's runtime deps (everything except numpy, which is ILP64 above)
RUN /databricks/python3/bin/pip install --no-cache-dir \
    psutil loguru threadpoolctl jax jaxlib jaxtyping typer progressbar2 bed-reader

# Jamma itself — --no-deps to preserve ILP64 numpy
RUN /databricks/python3/bin/pip install --no-cache-dir --no-deps \
    git+https://github.com/michael-denyer/jamma.git

# Verify ILP64 at build time — fail the image build if we got LP64
RUN /databricks/python3/bin/python -c "\
import numpy as np; \
cfg = np.show_config(mode='dicts'); \
blas = cfg['Build Dependencies']['blas']['name']; \
assert 'ilp64' in blas.lower(), f'Expected ILP64 BLAS, got: {blas}'; \
print(f'ILP64 verified: {blas}')"
