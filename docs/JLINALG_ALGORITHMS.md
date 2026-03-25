# jlinalg Algorithm Notes

jlinalg delegates all computation to vendor BLAS/LAPACK (MKL, Accelerate,
OpenBLAS) or NumPy fallback. The algorithms described below document what
vendor libraries implement internally, retained as reference for understanding
numerical behavior, performance characteristics, and debugging.

For architecture and file structure, see [JLINALG_ARCHITECTURE.md](JLINALG_ARCHITECTURE.md).

## 1. Goto/BLIS Three-Level Cache Blocking (vendor DGEMM/DSYRK)

*This section describes algorithms implemented by vendor BLAS libraries
(MKL, Accelerate, OpenBLAS, BLIS) that jlinalg dispatches to.*

### Why Blocking Matters

Naive matrix multiplication (`for i, j, k: C[i][j] += A[i][k] * B[k][j]`)
thrashes the CPU cache hierarchy. For an N x N multiply, the working set is
3N^2 doubles. At N=1000 that is 24 MB -- far exceeding L1 (32-64 KB) and L2
(256 KB - 1 MB). Cache misses dominate runtime, leaving the FPU idle.

The Goto/BLIS approach partitions the computation into three nested blocking
levels, each targeting a cache tier. Within each level, matrix data is
**packed** (copied into contiguous buffers in a microkernel-friendly layout)
so that the innermost computation accesses memory sequentially.

**Reference:** Goto & van de Geijn (2008), "Anatomy of high-performance
matrix multiplication", ACM TOMS.

### Three Blocking Levels

```
for jc = 0..N step NC:          # JC loop (L3: packed_B panel)
  for pc = 0..K step KC:        # PC loop (L1: packed_A column)
    Pack B[pc:pc+KC, jc:jc+NC]  # KC x NC -> packed_B (shared, L3-resident)
    for ic = 0..M step MC:      # IC loop (L2: packed_A row panel, OpenMP)
      Pack A[ic:ic+MC, pc:pc+KC]  # MC x KC -> packed_A[tid] (per-thread, L2-resident)
      for jr = 0..NC step NR:     # NR-wide column strips
        for ir = 0..MC step MR:   # MR-tall row strips
          Microkernel(MR, NR, KC)  # C[ir:ir+MR, jr:jr+NR] += packed_A * packed_B
```

| Level | Partition | Buffer | Cache Target | Size (AVX2) |
|-------|-----------|--------|--------------|-------------|
| JC | N into NC-wide panels | packed_B (KC x NC) | L3 | 8 MB |
| PC | K into KC-deep blocks | (repack B each iteration) | -- | -- |
| IC | M into MC-tall panels | packed_A (MC x KC, per-thread) | L2 | 144 KB |
| Micro | MR x NR tile | registers | L1 / registers | 12 KB (A column) |

### Packing

Packing copies a submatrix from its strided row-major layout into a
contiguous k-major panel format:

- **pack_A**: Copies an MC x KC submatrix into k-major order (k varies
  fastest, then row within MR-wide strips). Zero-pads the last strip if
  M is not a multiple of MR.

- **pack_B**: Copies a KC x NC submatrix into k-major order (k varies
  fastest, then column within NR-wide strips). Zero-pads the last strip if
  N is not a multiple of NR.

This layout ensures that the microkernel reads both packed_A and packed_B
sequentially, maximizing cache-line utilization and enabling hardware
prefetching.

### DSYRK Symmetry Optimization

Vendor `DSYRK` (K = X @ X.T) exploits symmetry: only the lower triangle is
computed, saving approximately 50% of floating-point operations compared to a
full DGEMM. When only the lower triangle is needed (e.g., for the eigensolver),
the mirror step can be skipped entirely.

## 2. Divide-and-Conquer Eigendecomposition (vendor DSYEVD)

*This section describes the algorithm implemented by vendor LAPACK's DSYEVD
routine. jlinalg dispatches to vendor DSYEVD (MKL, Accelerate) when available,
falling back to NumPy (`np.linalg.eigh`) otherwise. jlinalg does not contain
its own eigendecomposition implementation.*

The vendor DSYEVD algorithm uses a divide-and-conquer pipeline:

```
K  -->  DSYTRD  -->  (d, e, tau)  -->  DSTEDC  -->  (eigenvalues, Q)  -->  DORMTR  -->  eigenvectors
       tridiag       diagonal +        D&C on        eigenvalues of T      apply         eigenvectors
       reduction      off-diag +       tridiag        + eigenvectors       Householder   of original K
                     reflectors        matrix         of T                  reflectors
```

### Stage 1: Tridiagonal Reduction (DSYTRD)

Reduces the N x N symmetric matrix K to tridiagonal form T using Householder
reflectors:

```
K = Q_h * T * Q_h^T
```

where Q_h is the product of N-2 Householder reflectors and T has diagonal d
and off-diagonal e.

Vendor DSYTRD uses a **blocked algorithm** with block size NB (typically 64).
Each block applies NB Householder reflectors using an unblocked panel
factorization (`DSYTD2`), then performs a deferred trailing update via
`DSYR2K`:

```
A_trail -= V @ W.T + W @ V.T
```

where V contains the Householder vectors and W is a work matrix computed
from A and V.

**Reference:** LAPACK `dsytrd.f`, Dongarra et al. (1989).

### Stage 2: Tridiagonal Eigensolver (DSTEDC)

Solves the eigenvalue problem for the tridiagonal matrix T using
divide-and-conquer.

**Base case** (N <= 64): QR iteration with Wilkinson shift. The QR algorithm
applies similarity transformations to converge on eigenvalues. For small
matrices this is efficient and numerically stable.

**Recursive case** (N > 64): Split T into two halves T_1 and T_2 plus a
rank-1 correction:

```
T = [[T_1,  0 ],    +  rho * z * z^T
     [0,  T_2]]
```

where rho = e[m-1] (the off-diagonal element at the split point) and z is
constructed from the last row of Q_1 and first row of Q_2.

After recursively solving T_1 and T_2, the merged eigenvalues are found by
solving the **secular equation**:

```
f(lambda) = 1 + rho * sum_k( z_k^2 / (d_k - lambda) ) = 0
```

where d_k are the eigenvalues of the two subproblems and z_k are the
corresponding components of the rank-1 update vector.

### Secular Equation Solver (dlaed4)

The secular equation has exactly one root between each pair of consecutive
poles d_k. The solver uses the Gu-Eisenstat algorithm with several
refinements from LAPACK:

**ORGATI origin selection:** For each root, the solver selects the nearest
pole as the iteration origin. This ensures the delta vector (d_k - origin)
has full relative precision, avoiding catastrophic cancellation when poles
cluster.

**PSI/PHI split evaluation:** The secular function is split into two sums
around the target root:

```
PSI(lambda) = sum_{k < target} z_k^2 / (d_k - lambda)
PHI(lambda) = sum_{k >= target} z_k^2 / (d_k - lambda)
```

This split follows LAPACK's IIM1/IIP1 indexing for correct error bounds in
the rational interpolation step.

**Rational interpolation with Newton fallback:** The iteration uses a
rational function fit through three function values to predict the next
step. For clustered three-pole cases, `dlaed6` (cubic rational solver) is
dispatched. If the rational step fails validation, a Newton step is used as
fallback.

**Delta recomputation:** After convergence, deltas are recomputed as
`delta[k] = (d[k] - d[origin]) - tau` for maximum precision in the weight
product formula.

**Special cases:**
- N=1: trivial (single eigenvalue)
- N=2: `dlaed5` solves the 2x2 secular equation analytically using rho and
  z^2 formulas with a W-test branch for numerical stability

**Reference:** Gu & Eisenstat (1995), "A divide-and-conquer algorithm for
the symmetric tridiagonal eigenproblem", SIAM J. Matrix Anal. Appl.

### Eigenvector Computation (dlaed3 Weight Product)

After finding the eigenvalues, eigenvectors of the merged problem are
computed using the Gu-Eisenstat weight product formula:

```
v_j = z_j * product_{k != j}( delta_mat[j][k] / (d[k] - d[j]) )
```

where `delta_mat[j][k]` comes from the dlaed4 deltas. Critically, the
denominator uses `delta_mat[j][k] - delta_mat[j][j]` (not `d[k] - d[j]`)
because the deltas from dlaed4 have full relative precision, avoiding
catastrophic cancellation when eigenvalues cluster.

### Deflation

Near-degenerate eigenvalues (|d_i - d_j| < 8 * eps * ||T||_2) are deflated:
their eigenvectors are determined by Givens rotations rather than the secular
equation. This prevents the secular solver from encountering near-coincident
poles.

### Stage 3: Back-Transformation (DORMTR)

Applies the Householder reflectors from DSYTRD to transform the eigenvectors
of T back to eigenvectors of the original K:

```
U = Q_h * Q_dc
```

where Q_dc are the eigenvectors from DSTEDC and Q_h is the product of
Householder reflectors stored during DSYTRD.

Vendor DORMTR uses a blocked DLARFT+DLARFB algorithm, processing reflectors
in groups of NB (typically 64) from right to left. For each block, DLARFT
forms the upper triangular factor T encoding the product of reflectors, then
DLARFB applies the block reflector (I - V \* T \* V^T) to the eigenvector
matrix via DGEMM.

**Reference:** LAPACK `dormtr.f`, `dorgtr.f`.

## 3. Golub-Kahan SVD (vendor DGESVD)

*jlinalg dispatches SVD to vendor LAPACK (DGESVD) when available, falling back
to NumPy (`np.linalg.svd`) otherwise.*

The vendor routine implements the full Golub-Kahan bidiagonalization + QR
iteration algorithm.

For the JAMMA use case (LOCO eigenvalue update), the input is always
tall-skinny (m >= n). The computation proceeds:

1. **QR factorization** (dgeqrf): Reduces m x n matrix A to n x n upper
   triangular R via Householder reflectors: A = Q * R.

2. **SVD of R** (Golub-Kahan): The small n x n matrix R is bidiagonalized
   via Householder reflectors from both sides, then QR iteration with
   Wilkinson shifts converges on the singular values.

3. **Back-transformation**: U = Q @ U_R where U_R are the left singular
   vectors of R. This dgemm dominates the cost for tall matrices.

The `compute_uv=False` path skips step 3 and returns only the singular
values, which is faster when only the spectrum is needed.

**Reference:** Golub & Van Loan (2013), "Matrix Computations", Chapter 8.

### QR Factorization

Vendor LAPACK computes QR via DGEQRF + DORGQR. The factorization computes
A = Q * R where:
- Q is m x n with orthonormal columns (the "thin" Q)
- R is n x n upper triangular

Note that R must be extracted from the upper triangle of the DGEQRF output
BEFORE calling DORGQR, because DORGQR overwrites the Householder vectors
stored in the lower triangle.

When no vendor LAPACK is available, both QR and SVD fall back to NumPy
(`np.linalg.qr`, `np.linalg.svd`).

## References

1. Goto, K. & van de Geijn, R. (2008). "Anatomy of high-performance matrix
   multiplication." ACM Transactions on Mathematical Software, 34(3).

2. Gu, M. & Eisenstat, S.C. (1995). "A divide-and-conquer algorithm for the
   symmetric tridiagonal eigenproblem." SIAM J. Matrix Anal. Appl., 16(1).

3. Golub, G.H. & Van Loan, C.F. (2013). "Matrix Computations." 4th ed.,
   Johns Hopkins University Press.

4. Anderson, E. et al. (1999). "LAPACK Users' Guide." 3rd ed., SIAM.
   [Reference source](https://github.com/Reference-LAPACK/lapack).

5. Van Zee, F.G. & van de Geijn, R.A. (2015). "BLIS: A Framework for
   Rapidly Instantiating BLAS Functionality." ACM Transactions on
   Mathematical Software, 41(3).
