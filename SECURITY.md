# Security Policy

## Supported versions

JAMMA is published to PyPI from `master`. Security fixes are released against
the latest published version only; there are no long-term support branches.
Always run the most recent release.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue for a
suspected vulnerability.

- Use GitHub's [private vulnerability reporting](https://github.com/michael-denyer/jamma/security/advisories/new)
  ("Report a vulnerability" in the **Security** tab), or
- email the maintainer at <mdenyer@gmail.com> with `JAMMA SECURITY` in the subject.

Please include:

- the JAMMA version (`jamma --version`) and how it was installed (PyPI wheel,
  source, ILP64 numpy-mkl build);
- a minimal reproduction — ideally the input files (or a description of their
  shape) that trigger the issue;
- the observed behaviour (crash, hang, memory corruption, incorrect result) and
  what you expected.

You can expect an acknowledgement within a few days. Fixes are prioritised by
severity and coordinated via a GitHub security advisory.

## Scope and threat model

JAMMA is a numerical library for GWAS, not a network service. The relevant
attack surface is **untrusted input files** processed by the native code:

- the PLINK `.bed` / `.bim` / `.fam` readers, and
- the C extensions (`_lmm_accel`, `jlinalg`) that consume parsed genotype data.

Reports most likely to be in scope:

- memory-safety issues in the C extensions (buffer overflow, out-of-bounds
  read/write, integer overflow in size arithmetic) reachable from a crafted or
  malformed input file;
- denial of service from pathological inputs (unbounded allocation, hangs)
  triggered by file contents rather than by the operator's chosen parameters.

Generally **out of scope**:

- resource exhaustion from legitimately large but well-formed datasets — JAMMA
  is designed to consume the memory its inputs require (see the memory
  estimators and `docs/USER_GUIDE.md`);
- the optional runtime recompilation path, which invokes a compiler the
  operator already trusts on inputs the operator controls;
- vulnerabilities in third-party dependencies — report those upstream
  (JAMMA's lockfile is scanned daily by OSV).

If you are unsure whether something is in scope, report it anyway and we will
triage.
