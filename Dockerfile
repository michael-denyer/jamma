# Reproducible linux/amd64 JAMMA image with MKL-backed ILP64 NumPy.
# The build-capable stage compiles this checkout's native extensions; the
# runtime stage remains slim and receives only /usr/local from the builder.
FROM python:3.12.13-bookworm@sha256:3cd9086bdb30f7c9bc08a3fa621d9842e0d3f6f9291aeb4677e0547817c10b12 AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY docker/requirements-container.txt /tmp/requirements-container.txt
RUN python -m pip install --no-cache-dir --no-deps \
        -r /tmp/requirements-container.txt
# The ILP64 numpy wheel declares `mkl>=2026.1.0` and `mkl-service>=2.8.0`, and
# --no-deps means pip resolves neither, so both are pinned explicitly: `mkl` in
# the requirements file above, `mkl-service` here alongside numpy because it
# ships from the same custom index. Nothing in src/ imports mkl_service, which
# makes it look removable; it is not. Dropping it fails the ILP64 assertion
# below with "Intel oneMKL FATAL ERROR: Cannot load libmkl_avx2.so.3 or
# libmkl_def.so.3", because the MKL kernel libraries arrive with it.
#
# This numpy version must equal the [build-system].requires pin in
# pyproject.toml, so the extensions never build against newer headers than the
# runtime provides. test_dockerfile_provenance.py enforces that equality.
RUN python -m pip install --no-cache-dir --no-deps \
        numpy==2.4.6 \
        mkl-service==2.8.0 \
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

# One Intel OpenMP runtime per process: a second libiomp5 (for example one
# bundled in numpy.libs) crashes at interpreter exit.
RUN python -c "\
import jamma.lmm.accel, numpy as np; \
np.linalg.eigh(np.eye(64)); \
maps = open('/proc/self/maps').read().splitlines(); \
omp = sorted({l.split()[-1] for l in maps if ('libiomp' in l or 'libgomp' in l) and '/' in l}); \
assert len(omp) == 1, f'Expected one OpenMP runtime, got: {omp}'; \
print(f'OpenMP runtime: {omp[0]}')"

FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

COPY --from=build /usr/local /usr/local

RUN useradd -m -u 1000 jamma && mkdir -p /data && chown jamma:jamma /data
USER jamma

ENTRYPOINT ["jamma"]
CMD ["--help"]
