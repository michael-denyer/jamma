# Glossary

Terms and abbreviations used in JAMMA documentation.

| Term | Definition |
|------|-----------|
| **Accelerate** | Apple's BLAS/LAPACK framework, included with macOS. Provides ILP64 support on macOS 13.3+ via `$NEWLAPACK$ILP64` symbols. |
| **BLAS** | Basic Linear Algebra Subprograms -- standardized low-level routines for vector and matrix operations. JAMMA dispatches to vendor BLAS (MKL, Accelerate) for performance. |
| **DSYEVD** | LAPACK routine for symmetric eigendecomposition using divide-and-conquer. Faster than DSYEVR but requires O(N^2) workspace. JAMMA's default eigendecomp method. |
| **DSYEVR** | LAPACK routine for symmetric eigendecomposition using relatively robust representations. O(N) workspace -- used as fallback when DSYEVD workspace won't fit in memory. |
| **Eigendecomposition** | Factoring the kinship matrix K = U D U^T into eigenvectors (U) and eigenvalues (D). The most memory-intensive step in LMM GWAS. |
| **GWAS** | Genome-Wide Association Study -- testing each SNP for association with a phenotype while controlling for population structure. |
| **HWE** | Hardy-Weinberg Equilibrium -- a QC filter (`-hwe`) that removes SNPs deviating from expected genotype frequencies, indicating genotyping errors. |
| **ILP64** | Integer, Long, Pointer all 64-bit -- BLAS/LAPACK compiled with 64-bit integers. Required for matrices larger than ~46,000 × 46,000, since 46,000² ≈ 2.12 × 10⁹ elements overflows a 32-bit signed integer. |
| **Kinship matrix** | An N x N matrix measuring genetic relatedness between all pairs of N samples. Used by LMM to correct for population structure. |
| **LMM** | Linear Mixed Model -- a statistical model that accounts for both fixed effects (SNP, covariates) and random effects (kinship/relatedness) when testing for association. |
| **LOCO** | Leave-One-Chromosome-Out -- computing a separate kinship matrix for each chromosome, excluding that chromosome's SNPs. Avoids proximal contamination (the tested SNP influencing the kinship correction). |
| **LP64** | Long and Pointer 64-bit, Integer 32-bit -- standard BLAS/LAPACK with 32-bit integers. Limited to ~46k samples for eigendecomposition. |
| **LRT** | Likelihood Ratio Test -- an association test comparing the likelihood of the null model (no SNP effect) to the alternative model. JAMMA mode `-lmm 2`. |
| **MAF** | Minor Allele Frequency -- the frequency of the less common allele at a SNP locus. Used for QC filtering. |
| **MKL** | Intel Math Kernel Library -- highly optimized BLAS/LAPACK implementation. Best performance for JAMMA on Intel CPUs. |
| **Pab** | Projection matrix components used in GEMMA/JAMMA's REML optimization. Indexed using GEMMA's GetabIndex formula with 1-based indices. |
| **PVE** | Proportion of Variance Explained -- the fraction of phenotypic variance attributable to genetics (heritability estimate from the LMM). |
| **REML** | Restricted Maximum Likelihood -- the optimization method used to estimate the variance component ratio (lambda) in the LMM. More accurate than ML for small samples. |
| **Score test** | An association test based on the score function (gradient of the log-likelihood) at the null. Computationally cheaper than Wald/LRT. JAMMA mode `-lmm 3`. |
| **SNP** | Single Nucleotide Polymorphism -- a single-base genetic variant. The unit of association testing in GWAS. |
| **Wald test** | An association test based on the ratio of the estimated effect size to its standard error. The default and most common test. JAMMA mode `-lmm 1`. |
