# Changelog

All notable changes to JAMMA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- **The memory estimators are pure and the gate reads the machine once.**
  `estimate_streaming_memory` returns `MemoryLedger(kinship_gb, eigen_gb,
  lmm_gb)` with a `peak_gb` property, replacing the 13-field
  `StreamingMemoryBreakdown`, nine of whose fields nothing read.
  `estimate_lmm_memory` returns the batch phase's GB as a float, replacing
  `MemoryBreakdown`. Neither carries `available_gb` or `sufficient` any more:
  each gate reads `memory.available_ram_gb()` once and hands both figures to
  `memory.require`, so a test pins every decision, the eigendecomposition
  driver included, with one patch. `check_memory_available` is gone; its two
  callers use `require`. The margin is `memory.margin_gb`, public, and the
  only module that spells it; `eigen_plan`, `kinship/loco` and `lmm/eigen`
  import it. Every inequality keeps its exact form, LOCO's `<=` and margin-
  of-available included: `tests/test_memory_ledger_digest.py` pins 2,438
  estimates and gate decisions, tie rows included, as one sha256 recorded
  from `ebc07b6` before the code moved, priced under the NumPy dsyrk
  fallback so the digest is the same on every machine.

- **One compute entry point per kernel family, taking `lmm_mode` 1 or 4;
  `ABI_VERSION` 14 -> 15.** `compute_lmm_chunk_fused_c` and
  `compute_lmm_chunk_fused_general_c` now serve Wald and mode 4 both, reading
  the mode off the workspace they were handed. `compute_mode4_chunk_fused_c`
  and `compute_mode4_chunk_fused_general_c` are gone; callers pass the mode-4
  workspace to the surviving name instead. The returned dict carries the five
  Wald keys under `lmm_mode` 1, and those plus `p_scores`, `lambdas_mle` and
  `p_lrts` under 4, in the order the Python engine already reads them. Each
  merged function's per-SNP body is the mode-4 function's text unchanged, with
  the Score block and the LRT block under an `if (ws->mode == 4)` guard, so no
  arithmetic was reordered or re-associated. The n_cvt=1 Wald path now runs
  `coarse_grid_mode4_ncvt1_split` followed by `refine_lambda_ncvt1_split`,
  which performs the same REML operations in the same order as the
  `golden_section_lambda_ncvt1_split` forwarder it replaces; that forwarder had
  no other caller and is deleted. The Wald-only `output_arrays_t` and its
  alloc/decref/build triple are gone, and `mode4_output_t` is now `lmm_output_t`
  with `alloc_lmm_output(out, n_snps, with_mode4)`, `decref_lmm_output` and
  `build_lmm_result_dict`; the three mode-4 arrays stay NULL under `with_mode4`
  0 and the dict then omits their keys. Mode-4-only scratch (`thread_bufs` in
  the n_cvt=1 family) is allocated only under mode 4. Every compute-entry
  fingerprint key is bit-identical across the merge, all 87 shared keys over 94
  records, and `.assoc.txt` is byte-identical for all four `-lmm` modes at
  n_cvt 1, 2 and 4.
- **One general (n_cvt >= 2) workspace creator on the Pab table dict;
  `ABI_VERSION` 13 -> 14.** `create_workspace_general_c(eigenvalues,
  uab_invariant, UtW, Uty, n_samples, l_min, l_max, n_grid, n_refine,
  n_threads, pab_table, *, lmm_mode, hi_eval_null=None, logl_H0=None)`
  replaces `create_workspace_fused_general_c` and
  `create_workspace_mode4_fused_general_c`. `pab_table` is the dict
  `PabCTable._asdict()` returns, the same one `compute_*_split_general_c`
  already take, so `n_cvt` and the twelve index arguments are gone from the
  signature and `PabCTable.workspace_kwargs()` with them. `lmm_mode` is 1 or
  4 (Score-only and LRT-only at n_cvt >= 2 take no workspace), and mode 4
  requires `hi_eval_null` and `logl_H0`. `pab_table_t` carries
  `var_a_cols`/`var_b_cols`, and `parse_pab_table_from_dict` now performs
  every check the flat path did, including the four per-element range loops
  it lacked (`invariant_indices`, `varying_indices`, `logdet_diag_rows`,
  `logdet_diag_cols`) and the `int64` widening on the level-offset sum;
  `free_pab_table` zeroes the struct so a second call is a no-op. The two
  general compute entry points keep their names and check the workspace's
  `lmm_mode`. The Python wrappers `create_lmm_workspace_fused_general`,
  `create_lmm_workspace_mode4_fused_general`,
  `compute_wald_fused_general_c_ws` and `compute_mode4_fused_general_c_ws`
  are gone. Bit-identical on `.assoc.txt` for all four `-lmm` modes at n_cvt
  1, 2 and 4 and on every compute-entry fingerprint key.
- **One n_cvt=1 workspace creator in `_lmm_accel`; `ABI_VERSION` 12 -> 13.**
  `create_workspace_ncvt1_c(eigenvalues, uab_invariant, w, Uty, n_samples,
  l_min, l_max, n_grid, n_refine, *, lmm_mode, hi_eval_null=None,
  logl_H0=None)` replaces `create_workspace_fused_c`,
  `create_workspace_mode4_fused_c`, `create_workspace_score_fused_c` and
  `create_workspace_lrt_fused_c`. `lmm_mode` decides the null-model inputs
  (2 needs `logl_H0`, 3 needs `hi_eval_null`, 4 both, 1 neither) and the
  creator rejects an input the mode does not use. The four compute entry
  points keep their names and signatures; each now checks the workspace's
  `lmm_mode` instead of a per-family capsule name, so feeding a workspace to
  the wrong compute raises `ValueError` naming the mode. The four C structs
  behind those creators are one `lmm_workspace_t` (the LRT struct had no
  field the Wald struct lacked; the Score struct had two, `h_null_w` and
  `h_null_Uty`, which stay mode-3 only). The Score kernel's `(h*w)*x`
  association is untouched. The Python wrappers
  `create_lmm_workspace_fused`, `create_lmm_workspace_mode4_fused`,
  `compute_wald_fused_c_ws` and `compute_mode4_fused_c_ws` are gone;
  `chunk_kernel._ncvt1_kernel` binds the creator and the per-mode compute
  in one place. Bit-identical on `.assoc.txt` for all four `-lmm` modes at
  n_cvt 1, 2 and 4, and on every compute-entry fingerprint key.
- **`scripts/regenerate_fixture_manifest.py` is
  `scripts/check_fixture_manifest.py --write`.** The regenerator already
  imported every helper from the checker; `--write` is that second half.
  The manifest header and every message that named the old script now name
  the new command.
- **Five fixture generators are one `scripts/generate_gemma_fixtures.sh`.**
  `generate_all_gemma_fixtures.sh`, `generate_all_tests_reference.sh`,
  `generate_score_reference.sh`, `generate_covariate_reference.sh` and
  `generate_loco_fixtures.sh` are replaced by one script with an 18-row cell
  table, one `run_gemma` (local binary first, docker image with
  `--platform linux/amd64` second) and `--list`, `--dry-run`, `--only`. The
  LOCO kinship heredoc is `generate_loco_synthetic.py --loco-kinship`, with
  the subtraction formula unchanged; the old heredoc could not run on
  current code (it indexed `PlinkMetadata` as a dict and wrote `.npy` where
  GEMMA needed `.cXX.txt`). No committed fixture changed.

- **`gwas()` returns `PipelineResult`; `GWASResult` and `GWASTiming` are
  gone.** The API result was a copy of five of the runner's fields, so the
  runner's result is returned as is. `result.timing` keeps `kinship_s`,
  `lmm_s` and `total_s` and gains `load_s` and `rotation_s`; `assoc_path`,
  `assoc_paths` and `n_covariates` become reachable. `jamma.PipelineResult`
  is exported where `jamma.GWASResult` was.
- **`gwas(phenotype_column=2)` is `gwas(phenotype_columns=[2])`.** The
  keyword now takes the list `PipelineConfig` takes, so `[1, 2, 3]` runs
  three phenotypes against one eigendecomposition, as the CLI's `-n "1 2 3"`
  does. The other five knobs the CLI had and the API did not (`mem_budget`,
  `eigen_dir`, `n_grid`, `n_refine`, `legacy_text`) are added, and a test
  pins `gwas()`'s keyword set to `PipelineConfig`'s fields.
- **A bad option value exits 2, not 1.** The CLI builds one `PipelineConfig`
  and reports its `ValueError` as a usage error, so `-lmm 99`, `-maf 0.7`,
  `-hwe -1`, `-cat` without `-c`, and the like exit 2 with the config's
  message (`l_min must be positive` in place of `-lmin must be > 0`).
  Runtime failures (a missing file, insufficient memory) still exit 1. `-gk`
  is `click.IntRange(1, 2)`, so `-gk 3` is rejected by the option.
- **The hidden `-wsnp`, `-gxe`, `-vc`, `-mk` and `-mvlmm` stubs are gone.**
  They only ever raised "not yet implemented"; click now rejects them as
  unknown options. `-vc` is read as `-v -c`, which fails on the covariate
  file instead.
- **`PipelineResult.backend` is removed and the GEMMA log loses its
  `backend` line.** There is one backend. `log_backend_selection` takes
  `(requested, env_override)` only.
- **`PipelineRunner.parse_phenotypes` is removed.** `run()` reads the .fam
  once through `_load_phenotypes_and_intersect_masks` for every path,
  including LOCO, and `_parse_phenotype_column` requires the loaded
  `fam_data`.

### Fixed

- **The tier-marker gate no longer fails on a test file another worker is
  planting or removing.** `tests/test_conftest_c_seam.py` writes its
  transient `test_*.py` files under `tests/` through a temporary name the
  gate's glob does not match and publishes them with one `replace`, and
  `_file_untiered_functions` treats a file that vanished between the
  directory walk and the read as not part of the suite. A half-written or
  just-removed planted file had surfaced as `<unparsable file>` in about one
  CI run in ten.

### Changed

- **One reader for the "binary .npy default, GEMMA text legacy, .npy
  sidecar cache" contract.** `jamma.utils.npy_cache.read_array_artifact`
  loads a `.npy` path directly, a text path from its sidecar when the
  sidecar is at least as new and not corrupt, else parses the text and
  writes the sidecar; the caller supplies only the text parser, the shape
  check, and whether sidecar loads are memory-mapped. `read_kinship_matrix`
  and the eigen readers both call it, so kinship gains what only the eigen
  reader had: a truncated `.cXX.npy` sidecar is unlinked and the text
  re-parsed instead of raising, a text parse leaves a sidecar for the next
  read, and a read-only directory is a warning rather than an error.
  Kinship stays an eager, writable load because
  `apply_individual_weights` scales it in place; eigen sidecars stay
  read-only memory maps. `scripts/kinship_digest.py` 78/78 and
  `scripts/assoc_digest.py` 68/68 keys identical to `ace400c` (Mac).

- **The chunk engine carries only the thread budget it reads.**
  `_ChunkEngine.rot_threads` and `ThreadPlan.rot` are gone: the planner
  wrote them, the adaptive split rewrote them, and nothing consumed them
  (`_ChunkEngine.prepare` rotates under the process's BLAS thread count,
  not a per-engine budget). `omp_threads` stays because the kernel reads it
  on every chunk. Whether the rotation stage should hold its own
  `blas_threads` context is an open, unmeasured question, not something
  this change decides. `scripts/assoc_digest.py`: 68/68 keys identical to
  `18454df`.

- **LOCO plans its association once per run and prices it through the
  shared preflight.** `_run_lmm_for_chromosome_numpy` called
  `plan_association` for every chromosome (22 machine reads and dispatch
  logs per run) while the pipeline's own plan for the run reached only the
  banner and telemetry. The pipeline now plans LOCO with the new
  `loco=True` keyword, which selects a `loco` execution mode priced like
  streaming (one disk-read chunk plus the eigendecomposition, never the
  full genotype matrix LOCO does not hold), runs `memory_preflight` on it,
  and hands it to `run_lmm_loco(..., execution=)`; the chromosome loop
  carries one `LmmRunSpec` and relabels it per chromosome. `run_lmm_loco`
  called directly plans once itself, over the run's SNP total, and rejects
  a caller plan wider than `loco.col_chunk_size`. The banner and telemetry
  report `numpy-loco` for LOCO runs. `scripts/assoc_digest.py`: all 68
  keys identical to `19f7395`; `tests/test_memory_ledger_digest.py`
  unchanged.
- **The phenotype loop reaches the chunk engine through five signatures,
  not seven, and `-snps` is one filter.** `run_lmm_association_numpy_planned`
  and `run_lmm_association_numpy_streaming_planned` (one caller each, no
  tests) and the pipeline's `_run_batch`/`_run_streaming` pair are gone: the
  phenotype loop builds one `GenotypeSource` for the plan's mode and calls
  the shared body, now `run_lmm_association(source, spec, ...)`, which
  takes an `LmmRunSpec` (config, execution plan, `snps_indices`,
  `hwe_threshold`, `compute_pve`, `RunLabels`) in place of eight threaded
  keyword arguments. The batch path used to subset the genotype matrix
  before the runner while streaming passed the restriction as a filter;
  both now hand `snps_indices` to the body, where it joins the MAF,
  missingness and HWE filters. `RunInvariants.build` reads the prepared run
  and the config rather than fourteen re-listed fields. The public entries
  keep their signatures. `scripts/assoc_digest.py` (new, with the
  dispatchable `assoc-digest` workflow) records one sha256 per pipeline
  and API run over the synthetic and LOCO fixtures; all 68 keys are
  identical to `97c89b8`.
- **Every memory gate spells its inequality as `memory.fits`.** The DSYEVR
  fallback in `plan_eigen_driver`, the DSYEVD warning in `lmm/eigen`, and
  the LOCO single-pass decision each wrote their own comparison; two used
  `>` and LOCO used `<=`, so at an exact tie the eigen planner kept DSYEVD
  and LOCO went single-pass while `fits` said no. All three now call
  `fits`, which is strict, so the tie is a refusal everywhere. The LOCO
  multi-pass budget took `margin_gb(available_gb)` off the machine; it now
  subtracts the fixed costs from `memory.headroom_gb(available_gb)`, the
  inverse of `required + margin_gb(required)`, so the batch it returns is
  the largest one `fits` accepts. Below the 10GB cap that is a budget of
  `available / 1.1` instead of `0.9 * available`, so on small machines the
  batch can be one chromosome larger. `tests/test_memory_ledger_digest.py`
  is re-pinned: of 2,438 rows, 32 moved (12 `eigen:tie`, 18 `loco:tie`, and
  the 2 `loco` rows at n=10,001, 22 chromosomes, 8GB, batch 3 to 4);
  `scripts/dump_memory_ledger.py` dumps and diffs the table.

- **The chunk plan is one module.** `LmmChunkPlan.plan` and `LmmChunkPlan.narrow`
  replace the free functions `plan_lmm_chunks` and `tighten_lmm_chunks` and
  `ExecutableAssociationPlan.tighten_after_filter`; `_run_numpy_lmm` narrows
  `execution.conservative_chunks` directly. The planner is pure: `plan_association`
  resolves the per-chunk budget (`chunk_budget_bytes`) and the BLAS controllability
  once and passes them in, so `compute_chunk_size_numpy` now requires
  `mem_budget_bytes` and reads no RAM. Every plan is bit-identical to before;
  `tests/test_chunk_plan_digest.py` pins 7,560 plans over shapes, dispatch paths,
  budgets, caps and both BLAS states as one sha256 recorded from `8aa94f8`.
- **`n_refine` is raised to 20 in one place.** `LmmConfig` raises a lower value to
  `MIN_N_REFINE` at construction; the two downstream clamps in the chunk runner and
  `compute_lmm_chunk_numpy` are gone, and `DEFAULT_N_REFINE` is 20, the value every
  run already used. The preflight log now labels its chunk `pre-filter`, since it
  prices the conservative plan by design.

- **The 16-chunk cut for small inputs applies only under an uncontrollable
  BLAS.** `plan_lmm_chunks` consults `is_blas_controllable()`; with MKL or
  OpenBLAS the pipelined plan splits the cores and re-limits the thread pool
  per chunk, and on an 8-core Linux MKL node the cut measured +22.4% on the
  mouse_hs1940 shape (5 interleaved blocks, all positive), against -20% on
  an 18-core Apple M5 Pro with Accelerate. Linux keeps the plan it had
  before #294. On MKL the pipelined thread split also moves the rotation's
  last bits, so the cut is bit-neutral only under Accelerate.

- **The 16-chunk cut for small inputs stops at 10,000 samples.**
  `plan_lmm_chunks` applies `_PIPELINE_TARGET_CHUNKS` only up to
  `_PIPELINE_CUT_MAX_SAMPLES`. The interleaved A/B runner measured the cut at
  5,000 SNPs as -6.4% at 5,000 samples, a wash at 10,000, and +5.6% at
  30,000, where each extra chunk re-streams the 7.2 GB eigenvector matrix
  and the kernel it overlaps is a percent or two of the run.

- **The C accelerator computes logdet(H) as a mantissa product with an exact
  exponent.** `logdet_h_lambda` in the new `_lmm_logdet.h` splits each
  `lambda * ev + 1` into mantissa and exponent by its bit pattern, multiplies
  the mantissas in four lanes with renormalisation every 16 elements, adds the
  exponents as integers, and calls `log()` once. Every REML and MLE evaluation
  in both kernel families and both grid precomputes use it; the NumPy path
  keeps its per-element log sum. Results move only in the last bits (max
  relative difference 2.1e-14 against the log sum), so the fingerprint gate
  reports the compute entry points as changed by design, and no tolerance
  moved. `validate_eigenvalues` now rejects an eigenvalue that would make
  `l_max * ev + 1` non-positive, which the bit split requires. mouse_hs1940 on an
  18-core Apple M5 Pro, best-of-3: Wald 450 to 349 ms, all-tests 598 to
  384 ms, Wald with 4 covariates 878 to 813 ms; the n_cvt=1 kernel itself
  150 to 56 ms. See `docs/GEMMA_DIVERGENCES.md` section 3.

- **The C extension uses every physical core for OpenMP under Accelerate.**
  `get_c_extension_thread_count` no longer halves the count when threadpoolctl
  finds no controllable BLAS. The halving guarded against oversubscription in
  the overlapped pipeline, but a single-chunk run never computes beside a BLAS
  call, and the pipelined run still measured faster with the full count.
  mouse_hs1940 on an 18-core Apple M5 Pro, best-of-3: Wald 500 to 449 ms,
  all-tests 644 to 537 ms, Wald with 4 covariates 941 to 715 ms.

- **Small inputs pipeline rotation and compute.** `plan_lmm_chunks` cuts a
  split-capable run the memory budget alone would leave below
  `_MIN_PIPELINE_CHUNKS` to 16 chunks (`_PIPELINE_TARGET_CHUNKS`, floor 100
  SNPs), so the UT@G rotation of chunk N+1 overlaps the C kernel on chunk N
  instead of running first in one chunk. A plan the budget already splits past
  the threshold is unchanged. mouse_hs1940 on an 18-core Apple M5 Pro,
  best-of-3: Wald 520 to 400 ms, all-tests 630 to 536 ms, Wald with 4
  covariates 936 to 881 ms.

- **`_lmm_accel.c` marshals arguments through `_lmm_support.c`.**
  `take_vector`, `take_matrix`, `take_chunk` and `take_array` replace 33 of
  the 43 inline `PyArray_FROM_OTF` + shape-check + `INT_MAX` blocks;
  `validate_hi_eval_null`, `validate_logl_H0` and `validate_n_cvt` replace
  the four wordings of each guard; `build_grid_ncvt1` is the one n_cvt=1
  lambda grid, moved verbatim from the three identical copies. The two
  n_cvt=1 creators share `init_ncvt1_workspace` the way the general pair
  already shared theirs. Every workspace struct has one `*_free()` that the
  capsule destructor and every creator error path call, and every creator
  now creates its capsule before releasing the local array references. That
  ordering removes a double `Py_DECREF` in the Score and LRT creators'
  `PyCapsule_New` failure path, which released each input array once more
  than it owned. Error labels are `err_input`, `err_output` and `err_ws`.
  Bit-identical on the 79-key accel fingerprint and on `.assoc.txt` for all
  four `-lmm` modes at n_cvt 1, 2 and 4. `_lmm_accel.c` 4098 -> 3512 lines.
- **`_lmm_tests.{c,h}` are part of `_lmm_stats.{c,h}`.** The four
  Pab-to-statistic functions (`wald_from_pab`, `score_from_pab` and their
  general forms) sit beside the statistic-to-p-value ones they feed; both
  halves are pure double arithmetic that runs once per SNP after the
  optimizer. `LMM_ACCEL_SOURCES` drops one entry. Bit-identical on both
  digest levers.
- **Giant test files split along the src seams they mirror.**
  `test_runner_numpy.py` gives its chunk-sizing tests to a new
  `test_chunk_sizing.py` (absorbing `test_auto_tune_chunk.py` and
  `test_chunk_sizing_cap.py`), its dispatch-selection tests to
  `test_chunk_dispatch.py` and its results and output tests to
  `test_lmm_results.py`; `test_likelihood_numpy.py` gives its
  `compute_numpy` dispatch tests to `test_compute_numpy.py`;
  `test_loco_eigen_cache.py` gives its pure cache-key and manifest tests to
  `test_eigen_cache_key.py`, which no longer needs the mouse fixture. Pure
  moves; the set of test ids is unchanged.
- **One `BOUNDARY_SIZES` for the jlinalg BLAS tests** in `tests/builders.py`,
  with `EIGH_BOUNDARY_SIZES` capped at 200. The three copies were annotated
  with the MR/MC/KC of the own-BLAS kernel deleted at `663a22b`; the comment
  now says why the sizes are kept. `test_jlinalg_dgemm.py` sweeps 63, 64
  and 65 as well (three more parametrised cases). `TestDsyrkVendorDispatch`
  is deleted: its four tests repeated `TestDsyrkCorrectness`,
  `TestDsyrkSymmetry` and `TestDsyrkBoundary` at smaller sizes.
- **`tests/fakes/memory.py`.** `use_fake_psutil(monkeypatch, available=,
  total=, rss=, vms=)` pins both psutil reads JAMMA makes with
  `FakeVirtualMemory` and `FakeProcess`, replacing the 18
  `patch("jamma.core.memory.psutil.virtual_memory")` MagicMocks in
  `test_memory.py` and `test_eigendecomp_memory.py`; the fakes declare only
  the fields the code reads and their self-tests check those names against
  psutil's result types.
- **`jamma.io.read_fam_phenotypes(fam_path, column=1)`** and
  `parse_fam_phenotype_column(fam_data, column)` are the one `.fam`
  phenotype parser. `PipelineRunner._parse_phenotype_column` calls the
  latter; the test suite's own copy, `conftest.load_phenotypes_from_fam`,
  is deleted and its 28 callers read through `jamma.io`.
- **`tests/fixture_paths.py` and `tests/builders.py`.** Twenty-two test
  files each derived their own `fixtures` root and spelling of the mouse
  and synthetic paths, two carried verbatim copies of
  `NUMPY_GEMMA_TOLERANCES` and of `_build_snp_info`; the datasets are now
  frozen `FixtureDataset`s named once. `rotated_lmm_inputs` replaces the
  inline synthetic-input recipe in `test_likelihood_numpy.py` with
  bit-identical arrays, and `write_fam` replaces the ten `.fam` writers in
  `test_pipeline.py`.
- **`scripts/check_forbidden_patches.py` walks the AST.** The gate that bans
  patching numerical functions in tests matched line regexes, so a target
  wrapped by ruff or reached through `patch.object(cn, "name")` passed it;
  it now collects every patch call from the module's `ast`, resolves
  module-object arguments through the test's import table and
  canonicalises the target through the source tree's imports, so
  `patch("jamma.lmm.eigen.jlinalg.eigh")` is judged as `jamma.jlinalg.eigh`.
  It flagged 38 sites on the tree. Triage: the two `finite_difference_dev2`
  patches were vacuous (the reference `dev2` never calls it) and are gone;
  the `prepare_common` guards are driven by a negative and a NaN eigenvalue
  through the real optimiser instead of a `mock.Mock` one; `betainc`
  non-convergence starves `_CF_MAX_ITER` instead of replacing `_betainc_cf`;
  the `calc_pab`, `_check_symmetry_sampled` and
  `compute_iab_invariant_scalars_ncvt1` call-count tests are deleted; the
  golden-section evaluation count became an assertion that the returned
  `logl` is the REML at the returned `lambda`; eight dispatch spies that
  wrap the real function carry `# allow-patch:` with the reason.
- **`tests/fakes/jlinalg.py`.** `FakeJlinalg` replaces the six
  `patch("jamma.lmm.eigen.jlinalg")` MagicMocks, the `SimpleNamespace`
  fake in `test_safety_gates.py` and the triple patch in `test_lmm_unit.py`;
  `use_fake_jlinalg` is the one place a fake enters `eigen.py`. Its
  self-tests check the declared names against `jamma.jlinalg` and against
  every `jlinalg.<name>` read in `eigen.py`. The `eigendecompose_kinship`
  error-propagation and eigenvalue-zeroing tests moved from
  `test_jlinalg_eigh.py` to `test_lmm_unit.py`, five error tests
  parametrised into one.
- **`scripts/demonstrate_equivalence.py` uses `compare_assoc_results`.** The
  report kept its own field-diff code beside the validation package the
  tier1 suite uses; it now compares with `compare_assoc_results` and
  `compare_kinship_matrices` under the same `ToleranceConfig` overrides the
  tests pass, so `af` is compared on every section and `logl_H1` on the
  mouse `-lmm 4` fixtures. Its rank correlation is computed over
  `np.argsort`, so the script no longer imports scipy.
- **`scripts/_bench_common.py`** holds what `bench_all_backends.py` and
  `bench_loco.py` had each carried: the mouse fixture paths, `fmt_seconds`,
  `load_fam_phenotypes`, `add_gemma_args`, `find_gemma`,
  `print_hardware_header` and one `best_of` in place of six inlined timing
  loops. `bench_all_backends.py` runs as named phases over a `Timing`
  dataclass and `bench_loco.py` returns a `LocoTiming` in place of two
  differently shaped dicts. Flags and printed tables are unchanged.
- **The pre-push freshness check reads `LMM_ACCEL_SOURCES`.**
  `check_c_extension_freshness.py` watched its own `_lmm_*.c` glob beside
  the build's source tuple; it now loads `compile_and_link.py` by path and
  watches exactly the files the build compiles, plus the headers.
- **The CLI builds `PipelineConfig` once.** `_run_lmm` and `_run_gk` no
  longer mirror the config as 26 and 12 keyword parameters; `main` builds
  `OutputConfig` and `PipelineConfig` inside one `try` and hands them over.
  The five checks the CLI repeated from `PipelineConfig.__post_init__` and
  `LmmConfig` (`-lmin`, `-lmax`, duplicate `-n`, `-k` with `-loco`, the
  `--eigen-dir` default) are gone. `-n` and `-cat` share one parser, so
  `-cat '1,3'` works. `click.Path(path_type=Path)` replaces the CLI's own
  string-to-Path helper.
- **Config-only rules fail at construction.** `hwe_threshold`'s `[0, 1]`
  range, `-hwe` with `-loco`, `-cat` without `-c`, and a `-cat` column
  below 1 moved from `validate_inputs` to `PipelineConfig.__post_init__`,
  next to the other knob checks. `validate_inputs` keeps only what needs the
  filesystem.
- **`compute_kinship(config, mode: Literal[1, 2])` owns the `-gk` guards.**
  The mode range, `-eigen` with `-loco` and `-gk 2` with `-loco` are checked
  at its top, before any disk read, so a direct caller gets the CLI's errors.
  `_run_gk` no longer takes a phenotype column it never used.
- **`run()` and `_run_inner` are one method, and LOCO leaves below the shared
  preamble.** The LOCO branch used to re-read the .fam and load covariates on
  its own before the batch path did the same; it now returns after the
  shared covariate and phenotype loading, the valid mask and the dataset
  banner. The only visible difference is log order on `-loco` runs.
- **`load_kinship` returns K over `valid_indices` unconditionally.** It
  computes or loads the full matrix only when `save_kinship` needs a
  reusable file, writes it, and subsets afterwards; the caller no longer
  decides before and after the call. Same operations in the same order, so
  the saved matrix and the eigenpairs are unchanged.

- **`LocoConfig` has one `prefix` and no `save_kinship`.** Setting
  `kinship_output_dir` is what asks for each chromosome's K_loco to be
  written; leaving it `None` writes none, so the flag that could only agree
  with the directory or raise is gone, along with its `__post_init__` check.
  `kinship_output_prefix` and `eigen_prefix` always carried the same value
  (the pipeline passed `output_prefix` to both), so they are one `prefix`
  that also names the cache manifest. `PipelineRunner` maps its own
  `save_kinship` onto `kinship_output_dir` at the one call site, so the CLI
  and `gwas()` are unchanged. Direct `LocoConfig(...)` callers rename
  `eigen_prefix=` to `prefix=` and replace `save_kinship=True,
  kinship_output_dir=d` with `kinship_output_dir=d`.
- **`run_lmm_loco` no longer runs the eigen cache itself.**
  `loco_eigen.eigen_pairs_for()` decides once whether the run reads a
  validated cache or streams and eigendecomposes each chromosome, and owns
  the cache key, the directory, the manifest invalidate-then-commit, and the
  per-chromosome artifact writes. It returns an `EigenPairSource` (the pair
  iterator plus the PASS-1 SNP statistics, or `None` when the cache
  answered). `run_lmm_loco` shrinks from 349 to 207 lines and threads no
  Optionals through its loop. The manifest is still written only after the
  last chromosome's association completes, and an interrupted rewrite still
  leaves none. Every LOCO artifact (`.assoc.txt`, `.cXX`, `.eigenD`,
  `.eigenU`, manifest; binary and legacy text; cached, stale, partial and
  missing-phenotype runs) is byte-identical to before.
- **`eigen_io` reads and writes through one `_read_array` and one
  `_write_array`.** `read_eigenvalues`/`read_eigenvectors` were the same
  three-branch routine with the shape check inlined three times each, and
  the writers the same two-branch routine twice. The four per-array
  functions are now private; `read_eigen_files` and `write_eigen_files`,
  the two `jamma.lmm` already exported, are the API. Shape errors carry one
  wording per kind (`... has wrong shape (3, 2), expected a square matrix`).
- **`likelihood_numpy.py` is three modules.** `lmm/uab.py` holds the Uab,
  Pab and Iab batch builders in full, split and SoA layouts, which every
  path runs; `likelihood_numpy.py` keeps the NumPy-fallback grid and
  golden-section optimisers (1781 to 805 lines); `lmm/stats.py` holds
  `AssocResult` and the batch Wald, Score and LRT statistics that fill it.
  Pure moves, and the eight function-level imports of
  `classify_uab_columns`, `n_index`, `build_pab_table_for_c` and
  `reconstruct_uab_from_soa` are top-level imports, because `likelihood.py`
  imports nothing from `jamma` and the cycle they guarded against never
  existed. Callers rename `from jamma.lmm.likelihood_numpy import
  batch_compute_uab_numpy` to `from jamma.lmm.uab import ...`, and the
  batch `batch_calc_*_stats_numpy` / `_batch_lrt_pvalues_numpy` imports to
  `jamma.lmm.stats`.
- **One `reml_log_likelihood` and one `mle_log_likelihood`, keyed by
  `nc_total`.** The `_null` variants differed from the alternative-model
  ones by the number of columns projected out, so each pair is one function
  with a required keyword `nc_total` (`n_cvt` for the null model, `n_cvt +
  1` once the genotype joins). The GEMMA-literal scalar oracles with no
  production caller (`calc_wald_test`, `calc_score_test`, `calc_lrt_test`,
  `f_sf`, `safe_sqrt`, `calc_ppab`, `calc_pppab`, `reml_log_likelihood_dev2`,
  `logdet_hiw_null`) live in `tests/reference/{stats,likelihood}.py`;
  `finite_difference_dev2` stays, since `prepare_common` computes se(pve)
  with it. `likelihood.py` drops from 1216 to 895 lines.
- **`PabIndexTable` and `PabCTable` replace the two untyped table dicts.**
  `build_index_table` and `build_pab_table_for_c` return NamedTuples;
  consumers read `table.idx_yy`. `PabCTable._asdict()` is the dict every C
  entry point that takes a Pab table reads. The scalar `calc_pab`, the
  general `compute_Uab` and the REML
  logdet loop walk the same table the batch code walks: same integers, same
  arithmetic, same order.
- **`batch_compute_uab_numpy` takes `utg_t` of shape `(n_snps,
  n_samples)`**, the C-contiguous layout `jlinalg.dgemm(chunk, U,
  transa="T")` writes and its SoA sibling already took, so the chunk engine
  stops transposing. Callers holding `UtG` of shape `(n_samples, n_snps)`
  pass `UtG.T`. `compute_uab_invariant_soa` and `reconstruct_uab_from_soa`
  require `n_cvt`. `batch_compute_uab_varying_soa_numpy` validates `out`
  (shape, dtype, C-contiguity) once for every `n_cvt`; the n_cvt=1 branch
  used to check only the shape. `_NCVT1` names the six columns of the
  n_cvt=1 layout that the fast paths spelled as literals.
- **One negative-P_yy warn-once flag.** The scalar and batch guards each
  kept their own thread-local and reset function; `reset_p_yy_warned` and
  `warn_p_yy_once` in `likelihood.py` serve both. `chi2_sf(x)` drops the
  `df` parameter it only ever accepted as 1, and `special.py` vectorises
  `erfc` and `lgamma` through one helper.

- **The five `scripts/check_*.py` lints share one `_lint_common.py` and are
  named with underscores.** Each had carried its own copy of the same five
  steps — find the repo root, list the files, read one, decide whether an
  opt-out comment covers a finding, print the violations and pick an exit
  code — and the copies had drifted: two printed a different shape of report
  for the same kind of failure, and `repo_root = Path(__file__)...` appeared
  five times. `repo_root()`, `display_path()`, `tracked_files()`,
  `read_lines()`, `read_batch()`, `allowed()`, `report()` and
  `report_unreadable()` now live in one module and each lint keeps only its
  pattern table and its `scan_line`. The hyphens had to go because
  `check-quiet-flags.py` is not an importable module name, so there was
  nowhere to import that module from; the `.pre-commit-config.yaml` entries
  and the tests' `_SCRIPT` constants moved with them.

  The unreadable-file report that #171 added is preserved byte for byte.
  `read_lines` raises `LintReadError`, and `read_batch` is the one place that
  catches it, so a lint still ends with the header, the indented per-file
  line, and the footer saying that a skipped file is an unchecked file. The
  one wording change is that `check_test_timeouts` used to say "Test files"
  and "every test file"; the shared message says "Files" and "every target",
  because a single helper cannot hold two nouns for one idea. The tests now
  pin that header and footer rather than the exit code and a substring, which
  is what let a traceback pass for a report during this refactor.

  Two other behaviours changed. `check_quiet_flags` enumerates through
  `git ls-files` rather than `Path.glob` plus a seven-entry `SKIP_PREFIXES`,
  six of whose entries were a second copy of `.gitignore`; the two that were
  not (the lint and its own test, which carry the banned flags as fixture
  data) remain. `check_forbidden_patches` prints its findings in the same
  indented shape as the other four instead of separating them with blank
  lines, and its read failures now use the shared header instead of a
  bespoke "the gate is non-functional" one.

- **`check_doc_anchors.py` now states what a passing run does not prove.** For
  `docs/CODEMAP.md`'s tables the check has to work out which symbol an anchor
  means, and that is a guess: the rows label the link `file.py:123` and put the
  symbol in a separate column, so `_wanted_symbol` takes the first plausible
  backticked name in the row. A row naming two symbols can be checked against
  the wrong one. The gate is now under 94 anchors and `docs/DEVELOPMENT.md`
  advertised it by name without saying any of this, so a green run read as
  stronger than it is. Both the module docstring and DEVELOPMENT.md say so now.
  The fix, labelling each link with its own symbol, is about 120 links and has
  not been made.

- **`load_gemma_assoc` names both kinds of absent column.** Optional columns went
  through `_opt_float`, while beta and se used an inline
  `float(row.get(name, "nan"))` on the next line, two idioms for one idea. They
  are not one idea: an absent optional is `None` because the test does not report
  it, and an absent beta is NaN because `AssocResult` requires it and GEMMA's LRT
  formats omit it. Two named helpers, `_opt_float` and `_float_or_nan`, so the
  difference is stated rather than inferred from which idiom was used.

- **The `return_pab` flag is gone from the NumPy REML optimizers, along with all
  ten `@overload` stubs it needed.** Four private and public functions in
  `lmm/likelihood_numpy.py` returned either an array or a tuple depending on a
  boolean argument, so #124 added 128 lines of overload stubs to describe the
  union. Both production call sites in `compute_numpy.py` passed
  `return_pab=True`; the `False` arm of the two public optimizers was reached
  only by a test asserting its tuple length.

  `_batch_reml_at_lambda_numpy` computed `Pab_batch` unconditionally and
  discarded it when the flag was off, so returning it always is free. On the
  split-`n_cvt=1` path packing Pab does cost work, measured at 2.6 ms of 413 ms
  (0.6%) for 22 refinement iterations at 12,226 SNPs, which is not enough to
  justify a second code path.

  `_batch_golden_section_numpy` is now
  `_batch_golden_section_bracket_numpy` and returns the optimal log-lambda per
  SNP instead of evaluating there. Its three callers each own the final
  evaluation they actually want: the two REML optimizers take the Pab batch that
  falls out of it, and the MLE optimizer, which has no Pab, takes the
  log-likelihood alone. That removes the optional second evaluator parameter and
  the branch selecting between them.

  `likelihood_numpy.py` drops from 1990 to 1779 lines; `@overload` across `src/`
  drops from 18 to 8. No numerical change: the `.assoc.txt` output is
  byte-identical across `-lmm 1/2/3/4` at `n_cvt=1` and `n_cvt=4` on
  mouse_hs1940, with every C kernel disabled so the changed code is the code
  that runs.

- **`run_phenotype_loop` and `_run_batch` no longer take a kinship matrix.**
  `_run_inner` set `K = None` unconditionally on the line before the call, so the
  only value the parameter could ever hold was `None`, threaded through three
  signatures into `run_lmm_association_numpy(kinship=K)`. The pipeline consumes
  the kinship matrix during eigendecomposition and the runners take the
  eigenpairs, so there is nothing left to pass. #142 lifted this dead thread into
  a new module's public signature, which is where it became a contract rather
  than a stray local.

- **`get_plink_metadata` is read once per `-lmm` run instead of twice.** `run`
  read it for mode selection and `_run_inner` read it again. It parses the whole
  `.bim` (`sid`, `chromosome`, `bp_position` and both allele arrays), so the
  second call doubled that work. `run` now passes what it read down. Error
  ordering is unchanged, because the first read already preceded
  `validate_inputs`. The comment claiming the call "reads .fam/.bim header only"
  is corrected.

- **The `--hwe` rejection message has one home.** `run` and
  `_check_hwe_support` raised the same three-line `ValueError` text from two
  predicates. The predicates genuinely differ, since `run` fails before touching
  disk while `_check_hwe_support` re-checks the resolved plan after sample
  filtering, so both stay; the message is now a module constant they share.

- **The memory preflight gate moved to `pipeline_memory.py`.**
  `check_memory_requirements` and `_memory_preflight` were one question with two
  estimators, sitting 300 lines apart in `pipeline.py`, and read only
  `check_memory` and `mem_budget` off the config. They are now
  `check_streaming_memory(config, ...)` and `memory_preflight(config, ...)`,
  matching the shape `pipeline_kinship.compute_kinship` already uses. The two
  `MemoryError` messages each estimator raised are now built in one place, so the
  budget and insufficient-memory wording cannot drift between modes. `pipeline.py`
  goes from 1016 to 897 lines, back under the 1000-line bar the #142 split
  stopped just above.

  Tests that reached these as `PipelineRunner` methods now call the module
  functions. The batch estimator's monkeypatch target moves from
  `jamma.core.memory.estimate_lmm_memory` to
  `jamma.pipeline_memory.estimate_lmm_memory`: the old code imported it inside
  the function, so patching the defining module worked by accident of import
  placement.

- **`PipelineRunner._compute_valid_mask` is gone.** It was a `@staticmethod`
  whose whole body was a function-local import and a call to
  `prepare_common.compute_valid_mask`, so it held no runner state and read as
  though it did. Both call sites now call that function directly, which is what
  `loco.py` and `prepare_common.py` already do.

- **`compare_assoc_results` detects the reference test type as an enum instead of
  three booleans.** `is_all_tests`, `is_score_test` and `is_lrt_test` were derived
  from one sample of the reference rows, with Wald as the implicit `else` of an
  if/elif chain: an enum spelled as a boolean triple. It is now `AssocTestType`,
  produced by `_detect_assoc_test_type`.

  That matters because the four cases were re-derived in three separate places.
  #122 merged two of them and its comment claimed they shared one dispatch, but
  the SNP-count-mismatch early return still asked `is_score_test or is_all_tests`
  twice to decide which optional output slots to populate, and each of the four
  branches still ended in a hand-written conjunction of `.passed` attributes
  repeating overlapping subsets of the same columns.

  Two tables now hold the case analysis. `_OPTIONAL_SLOTS` says which optional
  result slots a test type populates, read by the early return. `_PASS_COLUMNS`
  says which columns the overall verdict reads, so the four conjunctions become
  one `all(...)`. LRT still omits beta and se, because it reports them as NaN by
  construction and verifies both sides are all-NaN instead; that is now one named
  branch rather than an inline special case inside a 240-line function.

  The per-type column selection stays a four-way branch. Those branches call
  different comparison helpers with different tolerances and skip messages, so
  there is nothing there to collapse.

  The file grows by 29 lines. The enum, the detector and the two tables cost more
  lines than the four conjunctions they replace; what improves is that the case
  analysis lives in three named places instead of being re-derived inline in
  three others.

- **The fixture-skip gate reads the source instead of watching skip reports.**
  #148 caught a fixture guard written as a skip by matching the exact phrase
  `fixture not available` in skip reports as they arrived, via
  `pytest_runtest_logreport` plus a module-level list plus a
  `pytest_sessionfinish` that mutated `session.exitstatus`. That could only fire
  when the guarded test actually ran, and only for that one wording. None of the
  suite's ~30 legitimate skips phrase anything that way, and neither would most
  new guards.

  `_enforce_no_fixture_skips` parses every `tests/**/test_*.py` at
  `pytest_configure` and rejects a `pytest.skip(...)` reason or a
  `@pytest.mark.skipif(..., reason=...)` whose text contains the word `fixture`,
  naming each file and line. Same mechanism as the tier-marker gate twelve lines
  above it, which is source-parsed for exactly this reason: it holds under xdist,
  `-k` and `-m`, and it flags the guard even in a file whose tests never ran.
  Only string literals are inspected, so a computed reason is left alone rather
  than guessed at.

  It immediately found one the old gate could not: `test_drift_is_detected` in
  `tests/test_fixture_manifest.py` skipped on "no tracked fixtures to
  drift-test against". `tests/fixtures/` is committed and manifest-listed, so an
  empty list means the checker stopped finding tracked files, which is the drift
  gate silently doing nothing. It asserts now.

  `require_fixture` is unchanged; it remains the mechanism tests call. Deleting
  the runtime plumbing removes the need to prove it under `-n 2`, so
  `tests/test_conftest_fixture_skip_gate.py` drops from 245 to 163 lines while
  covering more: five wordings the old phrase-match missed, plus `skipif`.

- **`check-doc-anchors.py` enumerates its inputs with `git ls-files`.** It
  carried a hand-copied `SKIP_DIRS`/`SKIP_FILES` pair mirroring
  `.markdownlint-cli2.jsonc`'s `ignores` and `lychee.toml`'s `exclude_path`, a
  third copy of one list with no way to read either other. Everything on it
  except `LICENSE.md` is gitignored, so git excludes it by construction.

  Coverage is a strict improvement, not merely equal: the same 94 anchors are
  checked, and `.pytest_cache/README.md` is no longer scanned. It is generated,
  and the hand-maintained list did not exclude it.

  The script now requires a git checkout and raises if `git ls-files` fails,
  rather than falling back to a filesystem walk that would quietly start
  checking uncommitted and vendored files. Its tests build a real repo and
  `git add` the doc, so they exercise the same enumeration the hook does.

## [7.2.0] - 2026-07-27

Minor, not major, despite the Breaking heading below. That is a deliberate
departure from strict SemVer and it is recorded here rather than left to be
inferred. All three breaks are on internal orchestration surfaces
(`PipelineConfig.phenotype_column`, when an out-of-range column is rejected,
and where `compute_kinship` lives). The two documented public entry points,
`gwas()` and the CLI, are unchanged: `gwas(phenotype_column=2)` still works and
every `-n` / `-gk` / `-lmm` flag behaves identically. Output is byte-identical on
mouse_hs1940 across `-lmm 1/2/3/4`, a two-phenotype run, LOCO, and five `-gk`
invocations. Given that, a major bump would cost every user a compatibility
review for a change none of them can observe.

### Breaking

- **`PipelineConfig.phenotype_column` is gone. Use `phenotype_columns`.** The
  config carried two fields for one concept and `__post_init__` kept them in
  step by mutating itself: an unset `phenotype_columns` was filled in from the
  scalar, then the scalar was overwritten with `phenotype_columns[0]`. A reader
  had to know that an empty list meant "unspecified" rather than "no columns",
  and that assigning either field after construction desynchronised them.
  `phenotype_columns` is now the only field, it defaults to `[1]`, and an empty
  list is rejected instead of silently meaning `[1]`.

  ```python
  # Before
  PipelineConfig(bfile=..., phenotype_column=2)
  # After
  PipelineConfig(bfile=..., phenotype_columns=[2])
  ```

  `gwas(phenotype_column=2)` is unchanged. It is a single-phenotype entry point
  (`GWASResult` has no per-phenotype output paths), so a scalar is the honest
  parameter there; it now maps to `phenotype_columns=[phenotype_column]`. The
  CLI's `-n` is unchanged.

- **An out-of-range phenotype column is now rejected when the config is
  built,** not part-way through `PipelineRunner.validate_inputs`. The rule had
  three homes: `validate_inputs` checked the scalar, `_parse_phenotype_column`
  re-checked whatever column it was handed, and `__post_init__` checked neither.
  It now lives only in `__post_init__`, which checks every index rather than
  just the first, so `phenotype_columns=[1, 0]` fails at construction instead of
  after kinship and eigendecomposition. Same `ValueError`, raised earlier. The
  message names `phenotype_columns`.

- **`PipelineRunner.compute_kinship()` moved to
  `jamma.pipeline_kinship.compute_kinship(config, mode)`.** `-gk` is a different
  program from `-lmm`: it shared nothing with the association pipeline but the
  config and the startup banner, and it returns a `KinshipResult` rather than a
  `PipelineResult`. It is a module function taking the config because
  `self.config` was the only instance state it read. `KinshipResult` is still
  importable from `jamma.pipeline`.

### Fixed

- **`IncrementalAssocWriter.write()` could raise `UnboundLocalError` instead of
  `OSError`.** `_write_buf` read its rollback position with `self._file.tell()`
  *inside* the try-block that protects the write. If `tell()` was what failed,
  `pos` was never bound, so the `except OSError` handler raised
  `UnboundLocalError` — which is not an `OSError`, so it bypassed the retry, the
  seek/truncate rollback, and the partial-file cleanup, and escaped the
  documented `Raises: OSError` contract with a misleading message. `tell()` now
  runs once before the loop. Reproduced with an `ESPIPE` from `tell()` and
  covered by `test_write_buf_tell_failure_surfaces_as_oserror`.

### Changed

- **`write_loco_kinship_matrices` accepts any iterable** of
  `(chromosome, kinship)` pairs, not only an `Iterator`. It walks the argument
  once and never calls `next()`, so a list or tuple was always valid at runtime
  and is now valid to the type checker too.
- **`get_hardware_context()` returns a `HardwareContext` TypedDict** rather than
  `dict[str, str | int | bool]`. Same eight keys, same runtime values — callers
  now get the per-key types the docstring already promised, so
  `ctx["cpu_count_physical"]` is an `int` instead of a union.
- **The `dgemm` type stub no longer claims float64-only operands.** Both the C
  binding and the NumPy fallback coerce `A` and `B` to float64, which
  `test_float32_coercion` has always asserted; the stub said otherwise. `out=`
  is unchanged — a non-float64 buffer is still rejected.

### Internal

- **`pipeline.py` is a third shorter, split along the seams the code already
  had.** 1476 lines down to 1016. Four groups moved out whole, each verified
  byte-identical on `mouse_hs1940` across `-lmm 1/2/3/4`, a two-phenotype run,
  LOCO, and five `-gk` invocations.

  | Moved to | What | Why it lifts cleanly |
  |---|---|---|
  | `pipeline_banner.py` | `log_dataset_banner`, `log_pipeline_banner` | Both were already `@staticmethod`, and nothing downstream reads what they print |
  | `pipeline_phenotype_loop.py` | `run_phenotype_loop`, `_run_batch`, `_run_streaming`, `PhenoLoopOutcome` | A closed subtree. `_run_inner` is the only entry and they call nothing else on the runner |
  | `pipeline_kinship.py` | `compute_kinship` | The `-gk` program, reached only from the CLI |
  | `jamma.io.snp_list` | `resolve_snp_list_file` (was `PipelineRunner._resolve_snp_list`) | A wrapper over the two functions in that module, now beside them |

  What made this checkable rather than a guess: `PipelineRunner.__init__` sets
  `self.config` and nothing else, so no method holds pipeline state and every
  extraction candidate was already a function of the config. The 628-line
  execution group was deliberately left whole. `_run_inner` calls thirteen
  siblings and everything under it is reachable only through it, so cutting
  there would put one program across two files.

- **Zero pyrefly errors, and the baseline is gone.** The committed
  `pyrefly-baseline.json` snapshotted 174 pre-existing errors so the gate could
  block new ones; it is now empty and has been deleted, along with the
  `--baseline` flag in the pre-commit hook and the CI lint job. The gate is
  absolute: any pyrefly error fails. A new one must be fixed, or given a narrow
  inline `# type: ignore[code]` at the offending line — visible in review in a
  way a baseline JSON diff is not.

  Grouping by root cause rather than by file is what made this tractable: 63
  entries across 24 files collapsed into nine causes, and one file's 23 entries
  turned out to be a single optional attribute. Along the way this removed a
  dead 28-line `@st.composite` strategy in `tests/test_hypothesis.py` that was
  defined once and referenced nowhere (vulture only scans `src/` and
  `scripts/`), and replaced `spec_from_file_location` in
  `tests/test_fixture_manifest.py` with the plain `sys.path` import the rest of
  the suite uses.

- **All three lychee entry points now pin the same binary version.** The
  pre-commit hook pinned `lychee-v0.23.0` while both workflows pinned only the
  `lycheeverse/lychee-action` SHA and inherited the action's own
  `lycheeVersion` default (`v0.24.2`), which is free to move in any future
  action release. Pinning the action does not pin the binary. All three are now
  explicitly on v0.24.2 — the version CI already resolved to, so CI behaviour
  is unchanged.

  The hook also needs `LYCHEE_VERSION=0.24.2` passed as its first argument.
  From 0.24.0 the hook script derives its version with
  `git describe --exact-match --match 'lychee-*v*'`, assuming it runs inside
  its own cached checkout — but `git commit` exports `GIT_DIR`, so the lookup
  lands in this repo instead, finds no `lychee-*` tag, and exits 100 on any
  commit that stages a markdown file. `prek run` sets no `GIT_DIR` and so does
  not reproduce it.

## [7.1.0] - 2026-07-26

### Security

- **Every zizmor finding fixed, and the gate tightened to match.** The
  pre-commit hook ran `--min-severity high` with three high findings
  ignored in `.zizmor.yml`. It now runs at default severity with no ignore
  rules, so medium and low findings block a commit. Fixes, not suppressions:

  - `persist-credentials: false` on all 16 `actions/checkout` steps
    (`artipacked`). No workflow pushes back to the repo, so none needed the
    token left in `.git/config`.
  - `issues: write` moved from workflow level to the single job that files a
    triage issue, in `flaky-detect.yml` and `sanitizers.yml`
    (`excessive-permissions`). The jobs that run tests and
    sanitizer-instrumented C now hold a read-only token.
    `link-check-external.yml` got the same shape for consistency.
  - `enable-cache: false` on `build-wheels.yml`'s sdist job
    (`cache-poisoning`), matching `publish.yml`. That job builds an artifact
    that gets published, and cache contents are user-controlled across runs.

  The `cache-poisoning` one was on a clock: the hook pins zizmor v1.24.1,
  which does not report it, while v1.25.x does — so the next dependabot rev
  bump would have failed the gate with no ignore covering it.

- **Dependabot `cooldown` on both ecosystems.** 7 days by default, 14 for
  semver-major on pip. A compromised release is usually yanked within days
  of publication, so waiting is the cheap half of supply-chain defence
  alongside the existing Action SHA pinning. Surfaced by the tightened gate
  (`dependabot-cooldown`), which the old high-severity bar never showed.

### Changed

- **refurb now gates every commit.** It shipped as `stages: [manual]` with
  ~23 pre-existing findings; those are fixed (30 by the time they were
  counted) and `stages: [manual]` is gone. Membership tests replace repeated
  comparisons, `contextlib.suppress` replaces `try/except/pass`, tuple
  literals replace throwaway lists in `for` targets, and a handful of
  single-use locals and an `else: return` are collapsed. No behaviour change.

  Three findings were **not** applied, with the reasons recorded in
  `[tool.refurb]` in `pyproject.toml` rather than as inline `# noqa`:

  - `FURB117` (`open(path)` → `path.open()`) is disabled outright. It is the
    same rule as ruff's `PTH123`, which `[tool.ruff.lint].ignore` already
    turns off as needless verbosity — so leaving refurb to demand it had the
    two linters contradicting each other on one question.
  - `FURB152` in `core/estimates.py` would replace `_EIGEN_ALPHA = 2.7152`
    with `math.e`. It is a fitted power-law exponent, not Euler's number;
    applying it would silently recalibrate every eigendecomp time estimate.
  - `FURB124` in `cli.py` would chain `gk is None and lmm is None` into
    `gk is lmm is None`, directly below its unchainable mirror image.

- **`LocoConfig` now owns the naming of every LOCO artifact.** The
  `.txt`-vs-`.npy` branch was written twice and the `{prefix}.loco.chr{chr}`
  convention three times, once per helper that built a filename.
  `artifact_suffix`, `eigen_stem()`, `eigen_paths()` and `kinship_path()`
  replace them, so the writer and the cache reader compose names from one
  place. `_computed_eigen_pairs` drops seven parameters as a result.

- **`save_kinship=True` without `kinship_output_dir` now raises** instead of
  silently writing nothing, matching `write_eigen`/`eigen_dir`.

  Released as a minor rather than a major bump. The behaviour it removes is a
  silent no-op: `_computed_eigen_pairs` tested `save_kinship and
  kinship_output_dir is not None` and skipped the write when the pair was
  half-set, so no caller could have been depending on it for output. The
  field's own docstring already called the directory required, and
  `LocoConfig` first shipped one release ago in 7.0.0, so the window in which
  anyone could have written that call is a single version.

- **`loco.py` split into three modules.** It had reached 1000 lines.
  `LocoConfig` and `DEFAULT_LOCO_CONFIG` move to `jamma.lmm.loco_config`, the
  eigenpair sources and artifact writers to `jamma.lmm.loco_eigen`, leaving
  `loco.py` as the orchestrator. `jamma.lmm.loco` re-exports both public
  names, so no import changes.

- **The LOCO pipeline branch builds its `LmmConfig` via
  `PipelineConfig.lmm_config()`** instead of writing the nine fields out by
  hand, which had made a second copy of that projection. `lmm_config()` takes
  `check_memory` as a keyword for the LOCO path, which returns before
  `_memory_preflight` and owns its own memory gate.

### Removed

- **Dead validation in `run_lmm_loco`.** Its own docstring already said
  invalid `lmm_mode` and `write_eigen` without `eigen_dir` are rejected when
  `LmmConfig` and `LocoConfig` are constructed; the body still checked both,
  with an error message that had drifted from `LocoConfig`'s. Two tests hid
  this by wrapping a call in `pytest.raises` where the raise actually fired on
  a config in the argument list.

- **Unreachable defaults on `_run_lmm_for_chromosome_numpy`.** It is private
  with one caller that passes all of them, and its `col_chunk_size = 5_000`
  was a second copy of `LocoConfig`'s default. Now keyword-only with `config`,
  `col_chunk_size` and `chr_name` required.

### Internal

- **pyrefly baseline down from 269 errors to 171.** Two root causes accounted
  for 98 of them: `test_telemetry.py` passed partial `BenchmarkRecord` dicts
  to a TypedDict whose five required keys `append_benchmark_record` never
  fills in (one error per missing key, hence 56 from 14 call sites), and three
  `tests/lmm_accel` modules called `Callable[..., Any] | None` C bindings
  without narrowing, behind runtime `skipif` gates a type checker cannot
  follow. Test-only; no shipped code changed.

  Note for anyone reading the tooling output: `pyrefly check --baseline`
  reports `0 errors (28 suppressed)`, and that count is inline suppressions —
  it appears identically with and without the baseline. Run without
  `--baseline` to see the real backlog.

## [7.0.0] - 2026-07-25

### Changed

- **BREAKING: `run_lmm_loco()` takes `LmmConfig` and `LocoConfig` instead of 23
  flat parameters.** Nine of those 23 were exactly `LmmConfig`'s fields, which
  6.0.0 had already made the only way to configure every other runner; LOCO was
  the last entry point still taking them loose. Ten more are LOCO-only and now
  live in the new `LocoConfig`; the remaining four (`bed_path`, `phenotypes`,
  `covariates`, `output_path`) stay direct parameters.

  ```python
  # before
  run_lmm_loco(bed_path=..., phenotypes=..., maf_threshold=0.05, lmm_mode=1,
               save_kinship=True, kinship_output_dir=d, show_progress=False)
  # after
  run_lmm_loco(bed_path=..., phenotypes=...,
               config=LmmConfig(maf_threshold=0.05, lmm_mode=1, show_progress=False),
               loco=LocoConfig(save_kinship=True, kinship_output_dir=d))
  ```

  `LocoConfig` validates at construction, so `write_eigen=True` without
  `eigen_dir` now fails where the config is built rather than where it is
  used — which matters when the two are far apart, as they are in the CLI.
  `run_lmm_loco` already rejected that pair at function entry, so nothing was
  ever eigendecomposed first either way. `col_chunk_size <= 0` is new.

  `LocoConfig` and `DEFAULT_LOCO_CONFIG` are exported from `jamma.lmm`.

  The private `_run_lmm_for_chromosome_numpy` drops from 22 parameters to 15
  by the same route.

### Removed

- **BREAKING: `jamma.lmm.run_lmm()` is gone.** It was a second dispatcher that
  routed pre-loaded arrays to the batch or streaming runner. `PipelineRunner`
  never used it — it calls `select_execution_mode()` and dispatches through its
  own `_run_batch`/`_run_streaming`, which also handle PLINK loading,
  incremental writing and timing — so the two routing paths had to be kept in
  step by hand for no benefit.

  Before removing it we checked: no callers in `src/` or `scripts/`, not in
  `jamma.__init__.__all__`, absent from the README and USER_GUIDE examples, and
  unreferenced by the one known downstream consumer.

  `select_execution_mode()` and `ExecutionPlan` are unchanged and still public.
  Programmatic callers should use `PipelineRunner`, or call
  `run_lmm_association_numpy()` / `run_lmm_association_numpy_streaming()`
  directly after picking a mode with `select_execution_mode()`.

### Documentation

- **Benchmarked v6.0.0 against v5.6.0 on mouse_hs1940.** Three interleaved
  rounds of best-of-3 per version, both built from clean worktrees with
  identical compiler flags and the same pinned numpy. Every operation lands
  within +/-2%, so the `LmmConfig` consolidation and the C translation-unit
  split cost nothing at small scale. `docs/PERFORMANCE.md` gains the comparison,
  the GEMMA 0.98.5 table, and reproduction steps.

- **Refreshed the README performance table** from Apple M2 to Apple M5 Pro on
  v6.0.0. The superseded M2 numbers are preserved in `docs/PERFORMANCE.md`. The
  currency note there now separates small-scale currency (current as of
  2026-07-25) from large-scale, which is still v4.2.0 and unchanged.

## [6.0.0] - 2026-07-25

### Changed

- **BREAKING: the LMM runner entry points take an `LmmConfig` and nothing
  else.** `run_lmm_association_numpy`, `run_lmm_association_numpy_streaming`,
  and `run_lmm` each accepted an optional `config` *and* nine flat keyword
  overrides of the same values, then unpacked one over the other in three
  separate places. Pass `config=LmmConfig(...)` instead of
  `maf_threshold=`, `miss_threshold=`, `l_min=`, `l_max=`, `n_grid=`,
  `n_refine=`, `check_memory=`, `show_progress=`, or `lmm_mode=`.
  `LmmConfig.as_kwargs()` existed only to service that round trip and is
  removed.

  This also closes a validation hole: `LmmConfig.__post_init__` bounds-checks
  every knob, but the flat path bypassed it entirely, so callers could pass a
  `maf_threshold` of 0.99 even though MAF is `min(af, 1-af)` and never exceeds
  0.5. Such calls now raise `ValueError`.

- **The p-value layer moved into its own translation unit.** `betainc_cf`,
  `betainc`, `f_to_pvalue` and the continued-fraction constants now live in
  `_lmm_stats.c`, with `chi2_sf_c` inline in `_lmm_stats.h`. `f_to_pvalue` was
  the most-referenced static in `_lmm_accel.c` (18 references from 12
  sections) and `chi2_sf_c` the third. Every remaining shared static is now an
  FP kernel or a workspace destructor. All 139 bit-exact fingerprint records
  are unchanged, and GEMMA parity is unaffected.
- **`PipelineRunner.validate_inputs` folds its seven file-existence checks
  into one ordered table**, 124 lines and 23 branches down to 104 and 17. The
  order in which the checks fire is part of the contract, since a config
  naming two missing files reports the earlier one, and it is now pinned by
  `tests/test_pipeline_validation_order.py`.

### Fixed

- **The NumPy `dsyrk` fallback no longer exceeds the memory the kinship
  pre-flight budgets.** It allocated a full N x N `np.dot` result plus the
  N^2/2 index arrays a whole-matrix mirror needs, none of it declared, so
  `_preflight_kinship_memory` could approve a run that then OOMed.
  Accumulation now walks the lower triangle in row blocks (halving the
  fallback's GEMM work as well as its peak) and the mirror tiles both axes.
  `jlinalg.dsyrk_scratch_bytes()` reports the bound and
  `StreamingMemoryBreakdown` gained `dsyrk_scratch_gb`.

- **The `_lmm_accel.c` coupling census counted things that were not
  references.** `scripts/lmm_accel_sections.py` reports which static functions
  are shared across section boundaries, and that list is the worklist for any
  extraction. It stripped a comment only when `/*` appeared on the same line,
  so the interior of every multi-line block comment was read as code: the
  file header alone manufactured cross-section references for
  `compute_lrt_batch_c`, `compute_score_batch_c` and `jamma_sentinel_oob`.
  Separately, the 616-line `PyMethodDef` block carried no section banner, so
  it was attributed to whichever section preceded it and every entry point
  registered there read as coupled to that one section. Real coupling was 24
  statics, not 58, and 28 of the difference was the method table alone.

### Internal

- The LMM dispatch path is resolved directly instead of deriving eight
  intermediate booleans and collapsing them through a priority ladder. The
  eleven `c_*_available` parameters became one `KernelCaps` record. Verified
  equivalent to the previous selector across all 24,576
  `(n_cvt, lmm_mode, availability)` combinations.
- LOCO chooses its eigen source once and iterates `(chr_name, eigenvalues, U)`,
  rather than testing "is the cache present" at six points inside the
  chromosome loop. `run_lmm_loco` drops from 476 lines and 57 branches to 399
  and 40.
- The batch and streaming runners share `prepare_lmm_run()` for the
  eigendecompose/rotate/null-model/PVE prologue they previously each inlined.
- Eighteen hand-written "C symbol is None" guards collapse into one helper.
  Six were `assert`, which `python -O` strips.

- **cppcheck now runs over `src/jamma/lmm`.** The hook was scoped to
  `^src/jamma/jlinalg/src/` from the day it was added, so the LMM accelerator
  had never been statically analysed. All five sources pass. `NPY_INTP_FMT`
  has to be defined for the run: undefined, cppcheck reports `unknownMacro`
  and stops analysing `_lmm_accel.c`, which then reads as clean because
  nothing was checked.
- **clang-format is explicitly disabled for `src/jamma/lmm`.** A grid search
  over 48 configurations found the best achievable diff was 22% of
  `_lmm_stats.c`, 48% of `_lmm_support.c` and 52% of `_lmm_accel.c`, with no
  single configuration suiting all three. The sources use manual column
  alignment the tool cannot reproduce. Without the opt-out the repo-root
  config applies, so running clang-format by hand rewrote 5,567 lines.
- 15 stale pyrefly baseline suppressions dropped, 289 entries to 272.

## [5.6.1] - 2026-07-21

### Fixed

- **Kinship-only runs are no longer refused for memory an eigendecomposition
  would have needed.** `-gk 1`/`-gk 2` write a kinship matrix and never
  eigendecompose, but their memory gate charged for
  `max(kinship, eigendecomp, lmm)` — roughly 3.7x the kinship phase's own
  footprint. A 50,000-sample kinship run needs ~24 GB and was rejected below
  80 GB; at 125,632 samples it needs ~136 GB and was rejected below 505 GB.
  The gate now sizes the kinship phase alone, matching how
  `eigendecompose_kinship` already gates its own allocation and how
  `PipelineRunner` already plans the whole workflow. Runs that genuinely do not
  fit are still refused. `StreamingMemoryBreakdown` gained `peak_kinship_gb`,
  which `estimate_streaming_memory` previously computed and discarded.

## [5.6.0] - 2026-07-21

### Added

- **`jlinalg.dsyrk` accepts a caller-owned output buffer.** The signature is now
  `dsyrk(X, *, out=None, beta=0.0)`, computing `out = X @ X.T + beta * out`. The
  existing one-argument call is unchanged. Batch, streaming, and LOCO kinship
  accumulation use `dsyrk(X, out=K, beta=1.0)` in place of `K += dsyrk(X)`,
  which removes the `n_samples x n_samples` temporary that the old pattern
  allocated on every batch.

### Changed

- **Mode 4 computes its REML and MLE coarse-grid brackets in one pass.** The two
  likelihoods share the same three SNP-varying reductions at each grid lambda
  and differ only in their tail, so the one-covariate split/SoA kernel now
  derives one canonical Pab per grid point and selects both brackets from it.
  Golden-section refinement and final evaluation remain independent per
  likelihood. Mode-4 output is bitwise unchanged. The general-covariate kernel
  is not affected.
- **`jlinalg` ABI version is 13.** The bump is internal to the C extension;
  ABI mismatches trigger the existing automatic recompile.

### Fixed

- **`dsyrk` no longer reads its output buffer when `beta` is zero.** BLAS
  defines `beta == 0` as "C is not read", precisely so the caller may pass
  uninitialized memory. For a zero-width `X` the native path scaled the freshly
  allocated output instead of zeroing it, and because `NaN * 0.0` is `NaN`, a
  reused allocation leaked its previous contents into the result. It is zeroed
  again.
- **`dsyrk` rejects an unaligned `out` instead of silently ignoring it.** An
  unaligned but otherwise valid buffer was converted to an aligned copy, so the
  result was written to the copy and the caller's array left untouched.
  `dsyrk(X, out=K) is K` now holds by construction.

## [5.5.0] - 2026-07-21

### Changed

- **LMM knob validation happens when `PipelineConfig` is constructed, not when
  the runner starts.** `PipelineConfig` re-declared the knobs `LmmConfig`
  already owned, so the rules for `lmm_mode`, `maf`, `miss`, `l_min` and
  `l_max` existed in three places and only fired part-way into a run.
  `PipelineConfig` now builds the `LmmConfig` its knobs imply — in
  `__post_init__` to validate, and via `lmm_config()` where the runners need
  it — so an invalid value raises immediately. Errors carry `LmmConfig`'s
  wording (`l_min must be positive` rather than `l_min must be > 0`). The
  CLI's own `-lmin`/`-lmax` messages are unchanged.

## [5.4.5] - 2026-07-21

### Fixed

- **Single-point lambda grids are rejected at the config boundary.** `LmmConfig`
  and `PipelineConfig` both accepted `n_grid=1`, though a one-point grid has no
  bracket to refine. The C kernel rejects it in `validate_batch_params`, but only
  once the run reaches the kernel — after kinship and eigendecomposition have
  been paid for — and the NumPy fallback has no such check, so it silently
  returned `lambda = l_min` for every SNP instead of its optimum. Both configs
  now raise `ValueError` on construction (`n_grid must be >= 2`), which also
  covers the LOCO branch, where `n_grid` is forwarded to `run_lmm_loco` without
  an `LmmConfig` ever being built (#78).

## [5.4.4] - 2026-07-20

### Fixed

- **Preflight false-OOM on multi-covariate runs.** The streaming memory
  preflight sized its compute chunk without `n_cvt`, then estimated Uab/Iab at
  the real `n_cvt`. Because Uab scales with `n_index = (n_cvt+3)(n_cvt+2)/2`,
  the estimated peak was inflated up to ~60× and rejected runs the runtime
  handles comfortably — a 25-covariate conditional analysis on 3,048 samples was
  estimated at ~467 GB and raised `MemoryError`, though the streaming runtime
  auto-sizes its chunk to `n_cvt` and completes in ~13 GB. Both preflight gates
  (`check_memory_requirements`, `check_memory_before_run`) now thread `n_cvt`
  into chunk sizing so the estimate uses the same chunk the runtime will (#74).

### Changed

- Dependency and CI-action bumps (Dependabot): `actions/checkout` 6 → 7,
  `pypa/cibuildwheel` 3 → 4, `actions/attest-build-provenance` 4.1.0 → 4.1.1,
  `j178/prek-action` 2.0.4 → 2.0.5, `astral-sh/setup-uv` 8.2.0 → 8.3.0, and dev
  dependencies `pyrefly` 1.0.0 → 1.1.1 and `hatchling` 1.30.1 → 1.31.0. No source
  or numerical-result change.

## [5.4.3] - 2026-07-13

### Security

- Bump `click` 8.3.1 → 8.4.2 in `uv.lock` to resolve PYSEC-2026-2132
  (CVSS 7.2, High), which the scheduled OSV Scanner flagged. No source
  changes; `click>=8.0.0` already permits the fixed release.

### Changed

- **Single source of truth for the eigen driver plan and LMM knobs.** The
  DSYEVD-inplace → DSYEVD → DSYEVR → numpy decision is centralised in
  `plan_eigen_driver`, shared by the runtime path and the pre-flight memory
  estimator so the two cannot drift; `run_lmm_loco` now honours the configured
  `n_grid`/`n_refine` instead of silently using hard-coded defaults; and the
  `DEFAULT_*` LMM knobs are defined once in `jamma.lmm.schema`. No API or
  numerical-result change.

### Fixed

- Corrected an inaccurate `DEFAULT_*` comment that claimed all knobs "match
  GEMMA v0.98.5" — `n_grid`/`n_refine` are JAMMA's golden-section parameters
  with no GEMMA equivalent (GEMMA uses Brent). Tidied the `EigenDriverPlan`
  type (dropped a dead field, typed `driver` as a `Literal`) and documented the
  `JLINALG_NO_VENDOR_LAPACK` presence-based contract and the `n_refine` min-20
  clamp. Internal only.

## [5.4.2] - 2026-07-09

### Changed

- **Shared NumPy LMM chunk runner**: batch, streaming, and LOCO paths now use
  `jamma.lmm.chunk_runner_numpy` for chunk sizing, rotation, C/Python dispatch,
  diagnostics, and per-chunk result writes.
- **Decomposed the shared chunk engine**: the former 1502-line
  `chunk_runner_numpy.py` is now a ~530-line orchestrator plus four focused
  sibling modules — `chunk_sizing.py` (RAM-budgeted chunk size),
  `chunk_workspaces.py` (persistent C-workspace lifecycle), `chunk_dispatch.py`
  (the C/Python kernel-selection ladder), and `chunk_pipeline.py` (thread split
  and overlapped pipeline). Each file is now well under 1000 lines. The batch
  runner's compatibility re-export shim was removed; callers and tests import
  each symbol from its canonical module.
- **Typed LMM dispatch path**: the per-run C-kernel choice is now a single
  `DispatchPath` enum (`jamma.lmm.dispatch`) instead of a bag of eight
  interdependent booleans on the former `LmmDispatch` dataclass. The chunk
  compute ladder and the persistent-workspace allocator now select the kernel
  with exhaustive `match` statements over one value, and the chunk runner's
  `feeds_raw_utg` / `uses_fused_score_or_lrt` predicates are properties on the
  enum. Illegal dispatch combinations are unrepresentable by construction, so
  the previous per-field validation is gone.
- **Deduplicated result sinks**: batch, streaming, and LOCO now build their
  per-chunk sinks via shared `make_writer_sink` / `make_result_list_sink`
  factories in `jamma.lmm.results` instead of inlining byte-identical closures.
- Removed the dead `fused_mode4` parameter from `compute_chunk_size_numpy`
  (the body never read it).
- **LOCO now runs through the shared C kernels at `n_refine=20`** (previously
  pure-NumPy at `n_refine=10`). Per-SNP `logl_H1`/`lambda` diagnostics shift
  slightly versus 5.4.1 as a result; p-values, effect sizes, and significance
  calls are unchanged, and GEMMA parity holds within the calibrated lambda
  tolerance.
- **Single source of truth for C-availability flags**: `chunk_sizing`,
  `chunk_workspaces`, and `chunk_runner_numpy` read the `_C_*` capability flags
  live from `compute_numpy` instead of snapshotting them at import, so
  sizing/workspace decisions cannot drift from the dispatch decision.

### Fixed

- **LOCO SNP statistics basis with missing phenotypes**: when some
  phenotypes/covariates are missing (analysed != all samples), LOCO now computes
  each chromosome's SNP mean/MAF, reported allele frequency, and missing-genotype
  imputation over the *analysed* samples — matching GEMMA (`src/lmm.cpp`, which
  averages and imputes over analysed individuals only) — instead of reusing the
  all-sample statistics cached during the kinship pass. The all-sample cache is
  still reused when every sample is analysed (identical result, no BED re-read).
  Previously the cached path and the non-cache / eigen-cache path could report
  different allele frequencies for the same run (and, with missing genotypes,
  different effect estimates).

### Removed

- Deleted the unused `write_streaming_chunk` helper from `jamma.lmm.results`
  (superseded by the shared per-chunk sink and inline diagnostics; only tests
  referenced it).

## [5.4.1] - 2026-06-10

### Changed

- **LOCO eigen-cache invalidation diagnostics**: `eigen_cache_is_valid` now
  reports a malformed manifest (parses but has no `cache_key`, e.g. an
  old-schema or truncated file) distinctly from a real input change, and
  enforces the manifest `schema_version` explicitly before the key compare
  instead of relying only on its presence in the hashed payload. A
  schema-version bump now invalidates all prior caches with a clear log reason.
  The manifest is `fsync`'d before its atomic rename so a crash cannot leave a
  parseable-but-garbage manifest, and corrupt-vs-unreadable manifests log
  distinct warnings. Manifest and components payloads are now typed via
  `TypedDict` (`EigenCacheComponents`, `EigenCacheManifest`).

## [5.4.0] - 2026-06-10

### Added

- **LOCO eigen-cache manifest**: per-chromosome eigen caches now carry a
  `<prefix>.loco.cache_manifest.json` manifest keyed by a SHA-256 over the
  inputs that determine the eigendecomposition: the `.bim` is content-hashed,
  the `.bed` is fingerprinted by size + modification time (not content-hashed,
  to avoid re-reading large genotype files), plus the MAF and missingness
  thresholds, `-ksnps` restriction, and the analysed-sample mask. On read, the
  key is recomputed and compared; a mismatch forces a full recompute. New
  module `jamma.lmm.eigen_cache`.

### Fixed

- **Silent stale LOCO eigen cache**: a cached eigendecomposition was reused
  whenever the per-chromosome files existed and matched on sample count, even
  if the SNP filters, `-ksnps` set, or analysed-sample subset had changed
  (same sample count, different missingness pattern). Those runs now detect the
  changed inputs via the manifest and recompute. A cache written before the
  manifest existed has no key; it is rejected and recomputed on every read run
  until regenerated with `-eigen`, which writes a manifest.

## [5.3.2] - 2026-06-03

### Added

- **Pyrefly type-check gate**: CI lint job and a prek hook now run
  `pyrefly check` against a committed baseline (`pyrefly-baseline.json`).
  The gate fails only on *new* type errors; the pre-existing backlog is
  snapshotted in the baseline and burns down over time
  (`uv run pyrefly check --baseline pyrefly-baseline.json --update-baseline`).
  Config lives in `[tool.pyrefly]` (pyproject.toml).

### Changed

- **Pyrefly gate hardening** (review follow-ups): `pyrefly` is now exact-pinned
  (`==1.0.0`) since the baseline is version-specific; the prek hook uses
  `always_run` so config/baseline-only edits also trigger it; the six C-ext-only
  Wald/Score/LRT wrappers now `assert <symbol> is not None` so pyrefly narrows
  them instead of suppressing via baseline (baseline 380 → 374); and a test pins
  the `_ACCEL_UNAVAILABLE` all-False/all-None invariant the `dict[str, Any]`
  sentinel build can no longer check statically.

### Fixed

- **Stale `_jlinalg.pyi` stub**: `dgemm` now declares its `out=` buffer
  parameter and `svd` its `compute_uv=` argument (via `@overload`), matching
  the compiled extension. Fixes false-positive type errors and IDE
  autocomplete for both hot-path BLAS calls.
- **Completed C-extension stubs**: `_jlinalg.pyi`'s `compute_snp_stats_chunk`
  had a stale 2-arg signature returning a tuple; corrected to the real
  preallocated-output form `(data, means, miss_counts, variances[, n_aa,
  n_ab, n_bb]) -> None`. Added the four `_lmm_accel.pyi` functions the stub
  omitted (`compute_score_batch_general_c`, `compute_lrt_batch_general_c`,
  `compute_score_split_c`, `compute_lrt_split_c`), verified against the
  compiled signatures and call sites. Drops the pyrefly baseline from 487 to
  454 errors.
- **`AccelImport` symbol typing**: the 30 C-function fields were typed
  `object | None`, so every guarded call site read as "not callable". Typed
  them `Callable[..., Any] | None`; the existing `is not None` guards now
  narrow to a callable. The all-unavailable sentinel is now built through a
  `dict[str, Any]` intermediate so its `**` spread type-checks against the
  callable fields (otherwise `dict.fromkeys(..., False)` flags `False` against
  them). Net clears 104 false-positive errors (baseline 454 to 380). Type-only
  change — annotations are PEP-563 strings, no runtime effect.

### Removed

- **`scripts/bench_secular.py`**: orphaned benchmark importing
  `jamma.lmm.loco_eigen_update`, a module removed with the secular solver in
  #68. The script could not run.

## [5.3.1] - 2026-06-03

### Added

- **CodeQL SAST**: weekly CodeQL static analysis workflow covering the C
  extensions and Python sources (#38).

### Changed

- **Internal refactors (no behavior change)**: extracted shared helpers and
  decomposed large methods across the LMM runners and pipeline. None of these
  alter CLI flags, output, or numerical results:
  - `validation/compare.py`: `.assoc.txt` parsing and comparison now derived
    from the output schema rather than hand-maintained column lists (434549f).
  - `lmm`: data-driven accel-symbol loader + shared streaming dispatch (#26);
    LOCO kinship streaming merged into the canonical kinship function (#27);
    shared `_create_workspaces` (#28), `_dispatch_compute` (#29), and
    `_drive_pipeline` (#30) extracted from both batch and streaming runners;
    per-thread scratch alloc/free helpers extracted in `_lmm_accel.c` (#33).
  - `pipeline.py`: `PipelineRunner._run_inner` god-method decomposed (#31);
    `-gk` kinship orchestration moved into `pipeline.compute_kinship` (#32).
- **Dependencies**: bump build-system `numpy` pin 2.4.4 → 2.4.5 (#24);
  bump `j178/prek-action` 2.0.3 → 2.0.4 (#23) and
  `google/osv-scanner-action` 2.3.5 → 2.3.8 (#22).
- **Test/CI hygiene (no behavior change)**: docstring, static-typing, and
  test-coverage follow-ups from the post-refactor review (#39); the
  batch/streaming FP-parity equivalence tests — bitwise-identical in optimized
  builds — now widen their tolerance only under the ASAN/UBSAN build, whose
  uninstrumented FP codegen drifts ~2e-10 on isolated elements (keyed on
  `JAMMA_SANITIZE`) (#39).

### Fixed

- **LOCO Python eigen-cache API**: `gwas(loco=True, write_eigen=True)` raised
  "write_eigen=True requires eigen_dir to be set" because the Python API never
  defaulted `eigen_dir` the way the CLI does. `PipelineConfig.__post_init__`
  now defaults `eigen_dir` to `output_dir` when `loco` and `write_eigen` are
  set; the README LOCO eigen-reuse example is corrected to the per-chromosome
  cache it actually produces (#37).
- **LOCO `--legacy-text`**: `--loco --legacy-text` silently wrote binary `.npy`
  instead of GEMMA-compatible `.txt`. `legacy_text` is now threaded through
  `run_lmm_loco` to the per-chromosome eigen-cache lookup, kinship save, and
  eigen write. Resolves `GEMMA_DIVERGENCES` §13 (#37).
- **LOCO multi-pass eigendecomp reserve sizing**: size the eigendecomposition
  workspace reserve by valid-sample count rather than the unfiltered sample
  count, so multi-pass LOCO batch sizing reflects the post-filter matrix
  dimension (#34).
- **Link check**: exclude canonical `gnu.org` license URLs from the lychee
  link check (`lychee.toml`) — gnu.org regularly times out for CI bots,
  producing spurious failures in the weekly external link check (#35).

## [5.3.0] - 2026-04-29

### Added

- **Weekly ASAN/UBSAN sanitizer workflow** (`.github/workflows/sanitizer.yml`):
  rebuilds C extensions with `-fsanitize=address,undefined` and runs the test
  suite under both sanitizers every Sunday. Uses a sentinel meta-test
  (`JAMMA_SENTINEL_UB=1` injects `-DJAMMA_SENTINEL_UB`, triggering a known
  heap-OOB) to verify the sanitizer harness is actually catching bugs and not
  silently passing. CI workflow runs with `set -o pipefail`, ASAN traces are
  written to a file rather than piped, and the sentinel asserter accepts both
  ASan heap-OOB and UBSan out-of-bounds signatures.
- **`JAMMA_SANITIZE` build seam** in `_build_support/compile_and_link.py`:
  appends `-fsanitize=...` flags and disables the post-link import probe so
  sanitized builds don't crash the wheel-build subprocess. Wired into all
  three compile entry points (`hatch_build.py`, `_compile_jlinalg.py`,
  `_compile_accel.py`). `check-compile-flag-literals.py` extended to recognise
  sanitizer flag literals so they are not flagged by the lint hook.
- **`JAMMA_FORCE_NUMPY_FALLBACK` env-var gate** for `jlinalg` and `lmm`:
  forces the NumPy fallback path even when vendor BLAS is available. Used by
  the sanitizer workflow to exercise the pure-Python paths and by debugging
  workflows where vendor-LAPACK output needs to be cross-checked against
  NumPy reference. Documented in `docs/TESTING.md` §1.10.
- **Sanitizer suppression file** for ASan and the heap-OOB sentinel
  (`-DJAMMA_SENTINEL_UB`). Documented in `docs/TESTING.md` §1.10 "Running
  under sanitizers (local repro of CI)".
- **New pre-commit hooks** (commit `2745846`): actionlint (GitHub Actions
  workflow lint), zizmor (workflow security audit), shellcheck (shell-script
  lint), vulture (dead-code detection), refurb (Python refactor suggestions),
  and `pytest-rerunfailures` (test re-run on transient failure). Three
  categories of pre-existing findings are deferred — see project notes for
  triage status and tightening conditions for each hook.
- **Tier-marker enforcement gate**: `tests/conftest.py` now AST-parses every
  collected test file in `pytest_configure` and fails the run when any file
  lacks a tier (`tier0`/`tier1`/`tier2`), `slow`, or `benchmark` marker. The
  gate runs once on the controller before xdist forks workers, fixing the
  silent fail-open under `-n N` that the previous collection-based gate had.
  Recognises parametrised markers (`@pytest.mark.skipif(...)`) and list-form
  `pytestmark`. Regression test `test_gate_fires_under_xdist` asserts the
  gate fires under `-n 2`.
- **Forbidden-patches gate**: new `scripts/check-forbidden-patches.py` +
  pre-commit hook bans patching `numpy.linalg.*`, `scipy.*`, and JAMMA's own
  numerical functions in tests. Feature-flag constants (`_C_*_AVAILABLE`)
  are excluded; `# allow-patch:` escape hatch documented. Now uses AST
  scanning rather than regex, covers `patch.object(<module>, ...)`,
  `mocker.patch(...)`, and `monkeypatch.setattr("dotted.path"...)`. Module-
  arg `monkeypatch.setattr(<module>, "<func>")` is also caught (closes a
  hole where two test files set callables on numerical modules and slipped
  past the previous gate). Read failures raise `_ScanError` and exit
  non-zero rather than passing vacuously on docs-only batches.
- **AST + runtime safety gates**: replaced regex source-greps in
  `TestLOCOIteratorRuntimeError` and `TestJlinalgABIValidation` with
  `ast.parse` structural checks plus runtime tests that exercise the
  guards (`python -O` subprocess for `loco_iter`; in-subprocess monkey-
  patched `_EXPECTED_JLINALG_ABI` for ABI drift, asserting on exit code
  and stderr).
- **Fakes package**: `tests/fakes/` provides `FakePipelineRunner`,
  `FakePipelineRunnerFactory`, `FakeAssocWriter`, `FakeProgressbarModule`,
  and `FakeProgressBar`. Type-narrowed to real `PipelineConfig` /
  `PipelineResult` so adding a required field actually breaks tests.
  `TestFakeProductionDrift` compares `inspect.signature` of each fake
  method to the real production method and fails with a specific drift
  message instead of silently masking new args. Adopted by `test_progress.py`
  (10 nested `patch(...) + MagicMock` blocks → one `fake_progressbar`
  fixture) and `test_cli.py` (4 `MagicMock` chains → one factory).
- **GEMMA fixture manifest**: `tests/fixtures/MANIFEST.toml` (55 entries)
  with SHA-256 of every git-tracked fixture. `scripts/check_fixture_manifest.py`
  verifies on-disk hashes match, flags untracked additions, and flags
  manifest-without-disk entries. `scripts/regenerate_fixture_manifest.py`
  rebuilds the manifest after intentional updates and auto-extracts
  `GEMMA Version` and `Command Line Input` from `.log.txt` headers.
  Pre-commit hook (fast) + tier0 self-test `tests/test_fixture_manifest.py`
  (slow) gate it.
- **Scheduled flaky-test detection**: `.github/workflows/flaky-detect.yml`
  runs the default suite under five distinct `pytest-randomly` seeds every
  Sunday 06:00 UTC. Non-blocking; opens an issue on disagreement.
- **Subsystem coverage gates**: per-subsystem coverage floors enforced in
  CI (`src/jamma/jlinalg/` floor at 18% to accommodate the Linux-vs-macOS
  vendor-LAPACK fallback delta — Linux measured 21.8% without MKL-ILP64,
  macOS-Accelerate measured 33.6%; both reference numbers documented in
  the threshold comment).

### Changed

- **Tier marker hygiene**: 8 previously-unmarked test files now have
  module-level `pytestmark`. `test_jlinalg_dispatch.py` converts
  `pytestmark = skipif(...)` to a list combining `tier0` + the existing
  `skipif`. `test_runner_numpy.py`: `:443`/`:518` GEMMA-parity tests
  promoted to tier1; `:396` internal dispatch test reclassified tier1 →
  tier0.
- **Tier3 marker removed** from `pyproject.toml`, both CI workflows,
  `conftest.py`, and both docs — defined and excluded everywhere but
  never used.
- **Scratch-bin renames** (git mv preserves history):
  `test_audit_fixes.py` → `test_lmm_audit.py`,
  `test_review_fixes.py` → `test_lmm_io_validation.py`,
  `test_loco_bugs.py` → `test_loco_orchestration.py`,
  `test_lmm_likelihood_dev2.py` → `test_likelihood_derivatives.py`.
- **Fakes drop call-count integers**: `FakeAssocWriter.call_count`,
  `FakePipelineRunner.run_calls`, `FakePipelineRunnerFactory.call_count`,
  `FakeProgressBar.start_calls`/`finish_calls` replaced with state
  booleans and lifecycle-violation `AssertionError`s. `update_calls:
  list[int]` retained because it records observable values, not counts.
- **`FakeProgressbarModule.widgets`** simplified from nested class to
  `SimpleNamespace(WidgetBase=_FakeWidget)`.
- **`test_jlinalg_lapack.py`**: folded `test_reconstruction_accuracy_large`
  and `test_orthogonality_large` into one
  `test_large_5000x200_reconstruction_and_orthogonality` (both checked
  the same 5000×200 QR — running it twice wasted CI minutes). Loosened
  orthogonality bound for the large case from 1e-14 to 1e-13 (theoretical
  floor for sqrt(5000) accumulation is ~1.6e-14).
- **`blas_backend` known-backends set** extended with `system-BLAS-ILP64`
  and `system-BLAS-LP64` (returned by `blas_dispatch.c:132` when a vendor
  library is loaded but path-string detection cannot identify it — typical
  on Linux distros linking against alias-only `libblas.so`).
- **`test_blas_backend_string_has_known_value`** asserts membership in a
  documented set (incl. `Accelerate-ILP64`) instead of printing.

### Fixed

- **Tier-marker gate failed open under xdist**: collection-based gate
  silently no-op'd whenever `-n N` was active (default `-n 3`). Empirically
  reproduced — an unmarked file ran cleanly under `-n 2`. Switched the
  gate to source-parsing in `pytest_configure` (runs once on the controller
  before xdist forks workers).
- **`monkeypatch.setattr(<module>, "<func>")`** previously bypassed the
  forbidden-patches policy. `test_lmm_accel.py:207` set
  `_compute_lmm_batch_c` to a sentinel and `test_prepare_common.py:282`
  set `_compute_score_batch_c` to `None` — both exited 0 under the old
  gate. Added a module-form rule keyed off the documented forbidden-module
  aliases (`compute_numpy`, `cn`, `likelihood`, `jlinalg`, `jl`,
  `kinship_compute`, `kc`), still allowing `_AVAILABLE`/`_ENABLED` flags.
  Audited the existing call sites and added `# allow-patch:` comments to
  the 5 legitimate dispatch toggles.
- **`scripts/check-forbidden-patches.py`** no longer swallows `OSError` /
  `UnicodeDecodeError`. Read failures now exit non-zero rather than silently
  producing zero findings (the silent-failure mode the gate is meant to
  prevent). Detects "argv passed but no `.py` among them" and falls back
  to a repo-wide scan with a stderr note instead of passing vacuously when
  pre-commit hands the hook a docs-only batch.
- **`tests/conftest.py`**: replaced silent `except ImportError: return` in
  `pytest_configure` with a stderr warning so a broken freshness script
  is visible.
- **`TestEigendecompLP64Threshold`**: replaced
  `contextlib.suppress(...)` with `pytest.raises(RuntimeError, match="test
  stub")`. The previous form could not distinguish "RuntimeError propagated
  to caller" from "caller silently caught and returned a default" — both
  passed the warning-routing assertion.
- **`.github/workflows/ci.yml`**: dropped `not tier3` from the default
  pytest filter (the marker was removed from `pyproject` / `conftest` /
  docs in `6d9ab15` but this one workflow line was missed).
- **`git mv` rename deletes**: the renames in `6d9ab15` staged the new
  files but the matching `D` entries for the old files were never added
  to the index, so the new files shipped alongside the old ones. Staged
  the deletes for `test_audit_fixes.py`, `test_review_fixes.py`,
  `test_loco_bugs.py`, and `test_lmm_likelihood_dev2.py`.
- **`tests/test_conftest_tier_gate.py`**: previously embedded a parallel
  stub of the old collection-based gate; after the xdist fail-open fix it
  was no longer testing the implementation it claimed to. Rewired the
  stub conftest to `importlib`-load the real `_enforce_tier_markers` from
  `tests/conftest.py`.
- **Removed dead `scripts/pre-push`**: standalone bash hook duplicated
  the `.pre-commit-config.yaml`'s `ruff-format-all` pre-push entry and
  was never wired into any git hook (`.git/hooks/pre-push` is prek-managed).

### Removed

- `tier3` pytest marker (defined but never used).
- `scripts/pre-push` (dead code; functionality lives in pre-commit).
- `docs/TESTING.md` §3.3 "Tests / markers to remove" (all rows were
  already done); subsequent sections renumbered.
- Stale 35-line "Test Tier System" block from `conftest.py` (claimed
  three tiers, listed nonexistent example tests, duplicated TESTING.md
  §1.5); replaced with a pointer to the source-of-truth doc.
- Three near-identical "@pytest.mark.slow on individual tests still
  applies" comments (restated standard pytest semantics).
- Transitional `FakeAssocWriter` re-export comment in
  `test_runner_numpy.py`.

## [5.2.1] - 2026-04-21

### Fixed

- Restore `#define _GNU_SOURCE` at the top of
  `src/jamma/jlinalg/src/blas_dispatch.c`. The BLIS strip in 5.2.0
  removed the define along with the `dladdr` scaffolding that
  originally motivated it, but two surviving `RTLD_DEFAULT` call sites
  silently relied on it too. `RTLD_DEFAULT` is exposed by glibc's
  `<dlfcn.h>` only under `_GNU_SOURCE`; the standard manylinux image
  happens to enable it via default CFLAGS, but the AVX2 manylinux
  image (gcc-toolset-14) does not — so 5.2.0 wheel builds failed on
  both Linux jobs and no wheels reached PyPI. 5.2.0 should be
  considered unreleased; install 5.2.1 directly.

## [5.2.0] - 2026-04-21

### Added

- **Build-support consolidation**: new internal `jamma._build_support`
  package (`compile_and_link.py`, `openmp_detect.py`, `find_compiler.py`)
  is the single source of truth for compile flags, source lists, and
  link flags used by `hatch_build.py` (PEP 517 wheel path),
  `_compile_jlinalg.py` and `_compile_accel.py` (dev-mode and runtime
  recompile entry points), and the `jamma.core.recompile` ABI-mismatch
  shim. Every bare compile flag (`-O3`, `-fno-fast-math`, `-fopenmp`,
  etc.) now lives in one file; two pre-commit hooks
  (`check-compile-flag-literals.py`, `verify_compile_invocations_match.py`)
  enforce this.
- **Runtime recompile hardening**: new `jamma.core.recompile` shim uses
  a file-lock + atomic `os.replace` to serialize concurrent recompiles
  (pytest-xdist workers, parallel Databricks jobs, multiple notebook
  kernels) so they no longer race on the same `.so` path and produce a
  corrupted file. The `_compile_accel` path now verifies the freshly
  compiled `.so` actually imports before returning success — a missing
  export or bad RPATH previously let the recompile report success with
  an unusable extension.
- **Stale C extension drift detection**: new `check_c_extension_freshness.py`
  pre-push hook detects when a committed `.so` is older than its source,
  preventing pushes that would ship stale binaries.
- **CI/lint discipline**: new pre-commit hooks `check-quiet-flags.py`
  (bans `-q` / `--silent` / `--quiet` and pre-commit skip flags in
  committed code), `check-test-timeouts.py` (flags unjustified long
  pytest timeouts), and `ruff BLE001` (bans blind `except Exception`).
  New `package-smoke` CI job inspects sdist + wheel contents to prevent
  missing `_build_support` files from shipping.

### Changed

- **Pipeline refactor**: `PipelineRunner._run_inner` split into
  `_memory_preflight`, `_load_phenotypes_and_intersect_masks`, and
  `_run_loco` helpers. Shared LMM compute helpers promoted to public
  names (`build_uab_tab`, `_build_results`, etc.) to support the
  extracted dispatch-path selector.
- **LMM dispatch extracted** from `run_lmm_association_numpy` into
  `src/jamma/lmm/dispatch.py` — the ~60-line logic for selecting
  between fused/split/general kernels by `n_cvt × lmm_mode ×
  kernel-availability` is now independently unit-testable.
- **OpenMP downgrade visibility**: runtime recompile retries that fall
  back to single-threaded execution now surface a `warnings.warn()`
  rather than disappearing silently (closes the gap between build-time
  and runtime diagnostics).
- **Documentation**: WHY_JAMMA tolerance table disambiguates golden-section
  vs Brent optimizer attribution. Mermaid diagrams across
  `README.md`/`docs/` migrated from literal `\n` (which renders as
  backslash-n) to `<br/>` for proper line breaks. Build-plumbing
  references refreshed to match the `_build_support` consolidation.
- **CI**: Node runtime bumped from 22 to 24. `michael-denyer/numpy-mkl`
  references bumped to 2.4.4. Dependabot SHA-pinning now covers
  `github-actions` ecosystem.

### Fixed

- **AccelImport retry-path drift** (latent bug): the post-auto-recompile
  unpack in `compute_numpy.py` had 33 targets vs the 35-field
  `AccelImport` NamedTuple, missing `compute_score_split_general_c` and
  `compute_lrt_split_general_c`. A successful runtime recompile would
  have raised `ValueError: too many values to unpack` instead of
  recovering. Both unpack sites replaced with field-by-field binds so
  they cannot drift.
- **`_compile_accel` reported false success** (latent bug): returned
  True on compile+link without verifying the produced `.so` imports,
  so bad RPATH / missing runtime lib / ABI mismatch let
  `python -m jamma.lmm._compile_accel` exit 0 and `auto_recompile`
  report success while the real `import` still raised. Import
  verification re-added (mirrors `_compile_jlinalg`).
- **`jlinalg` recompile diagnostics invisible on Databricks**: replaced
  two `print(..., file=sys.stderr)` blocks with `warnings.warn()` so
  recompile-skipped / recompile-but-import-failed messages route
  through the same channel as the surrounding `warnings` and aren't
  swallowed by notebook stderr capture.
- **`_build_support/__init__.py` docstring** described a non-existent
  `sys.path.insert` loader; rewritten to match the actual
  `importlib.util.spec_from_file_location` + `jamma_build_support.*`
  namespace mechanism used by `hatch_build.py`.
- Runtime recompile lock-file paths (`*.so.lock`) now gitignored to
  prevent accidental commits.
- Batch LMM memory preflight threads `n_cvt` through
  `check_memory_before_run` so multi-covariate runs don't silently pass
  a single-covariate preflight and OOM at the real allocation.

### Removed

- **BLIS dispatch path** from `src/jamma/jlinalg/src/blas_dispatch.c`.
  The `discover_bundled_blis()` discovery routine, the `is_blis`
  parameter threaded through six resolver functions, and the
  co-located `libblis-firestorm.dylib` binary (never tracked in git,
  never shipped in any wheel) are gone. jlinalg now dispatches to
  vendor ILP64 BLAS/LAPACK (Accelerate on macOS 13.3+, MKL-ILP64 on
  Linux/Windows via the `michael-denyer/numpy-mkl` index) with NumPy
  fallback otherwise — no middle tier. BLIS was BLAS-only; eigh fell
  through to NumPy anyway, so the dispatch path offered no net speedup
  on any active install. Net: `-184 / +49` lines in `blas_dispatch.c`,
  plus related cleanup across `jlinalg.h`, two tests, and two core
  docstrings.
- Dead LP64 branch in `jlinalg.select_best_backend` and stale legacy
  fields from `jlinalg_eigh_status_t` — jlinalg was never wiring LP64
  backends anyway; the dead code inflated the API surface.
- Orphaned `_compile_utils.py` and legacy `openmp_detect.py` in
  `jamma.core` (moved to `jamma._build_support`).
- Redundant `auto_recompile` re-export shim in `jamma.lmm`.

## [5.1.6] - 2026-04-15

### Fixed

- Batch LMM memory preflight now propagates `n_cvt` to `estimate_lmm_memory`
  at both call sites (`PipelineRunner._run_inner` batch branch and
  `run_lmm_association_numpy`). Previously these passed only
  `(n_samples, n_snps)`, silently defaulting `n_cvt=1`, so multi-covariate
  runs could pass the preflight and then OOM at the real `Uab_batch` /
  `Iab_batch` allocations (which scale with `n_cvt`). The streaming branch
  was already correct.

## [5.1.5] - 2026-04-11

### Added

- Warn when fewer than 50 samples enter the LMM (after phenotype/covariate
  filtering). LMM-based GWAS has insufficient statistical power below this
  scale, and JAMMA's batch golden-section lambda optimizer assumes unimodal
  log-likelihoods — an assumption most likely to fail at very small n. The
  warning fires once per run from both the pipeline (CLI) and `run_lmm()`
  (programmatic API). See `docs/GEMMA_DIVERGENCES.md` §6.

### Changed

- Rename `EQUIVALENCE.md` → `GEMMA_EQUIVALENCE.md` and
  `NUMERICAL_EQUIVALENCE_BOUND.md` → `GEMMA_NUMERICAL_EQUIVALENCE_BOUND.md`;
  update all cross-references across docs, tests, README, and CHANGELOG
- Link previously orphaned `GEMMA_NUMERICAL_EQUIVALENCE_BOUND.md` from README

## [5.1.4] - 2026-04-08

### Changed

- Remove 7 `inspect.getsource()` anti-pattern tests and rewrite
  `test_lapack_no_ffast_math` to parse build config files as text
- Replace `MagicMock` with real types and fakes across test suite
- Add pre-commit hook banning `inspect.getsource()` in tests
- Add test type routing and bug fix workflow sections to TESTING.md

## [5.1.3] - 2026-04-07

### Fixed

- Compiler detection now uses cc/clang/gcc fallback chain instead of failing
  when `CC` is unset or points to a missing compiler
- `hatch_build.py` uses the same fallback chain for wheel builds
- Narrow exception catches in `_compile_jlinalg.py` — no longer swallows
  unexpected errors during C extension compilation
- Assert C extension is loaded in CI to catch silent compilation failures

### Added

- Sigstore build provenance attestations on PyPI publish
- OSV vulnerability scanning on pull requests
- YAML-form issue templates (bug report, feature request)
- Streaming covariate integration tests

### Changed

- Replace pre-commit with prek (Rust-based, no Python dependency)
- Pin all GitHub Actions to commit SHAs (Dependabot keeps them updated)
- Pin `hatchling==1.29.0` and `numpy==2.4.3` in build-system.requires
- Use `--index-url` instead of `--extra-index-url` for custom package indexes

### Security

- Harden supply chain: pinned actions, Sigstore attestations, osv-scanner
- Dependabot configured for GitHub Actions ecosystem (weekly)

## [5.1.2] - 2026-04-02

### Fixed

- Ctrl+C during eigendecomposition now exits immediately instead of blocking
  until the LAPACK call finishes
- Progress bar no longer shows 100% before propagating worker exceptions
  (MemoryError, LinAlgError)
- Broken pipe on stdout no longer masks eigendecomposition results
- Remove meaningless AdaptiveETA widget from time-based progress bar

### Added

- Tests for `timed_progress()`: exception propagation, 99% cap, error display,
  `estimated_seconds=0` edge case

## [5.1.1] - 2026-04-01

### Fixed

- Time estimates now show BLAS backend caveat when not running on MKL
  (estimates are calibrated to MKL ILP64 on 48-core Xeon)
- Memory pre-flight check logs active BLAS backend and ILP64 status; warns
  when >40k samples without ILP64 or when time estimates are uncalibrated
- Fix pip install order in docs: deps first, numpy-mkl second, jamma --no-deps
  last to prevent ILP64 overwrite

### Changed

- High contrast mermaid diagrams across all docs (README, CODEMAP,
  JLINALG_ARCHITECTURE, USER_GUIDE) with dark subgraph backgrounds and
  bright node fills
- Add three new diagrams to USER_GUIDE: GWAS pipeline flow, BLAS/eigendecomp
  dispatch, and memory safety architecture

## [5.1.0] - 2026-03-25

### Added

- Telemetry transparency: opt-out via `JAMMA_NO_TELEMETRY=1` or `DO_NOT_TRACK=1`,
  with docs and hardening for privacy-sensitive environments
- Safety gates for LP64 integer overflow, LOCO chromosome invariant, and ABI
  validation at import time
- GEMMA equivalence tests for full validation coverage

### Changed

- Rename pipeline methods to `_run_batch`/`_run_streaming` for clarity
- Remove dead `lmm_mode` parameter from `select_execution_mode`
- Remove dead backend dispatch types and simplify consumers
- Consolidate dev dependencies and clean up CI build matrix
- Fix incorrect `gwas()` API examples in README and USER_GUIDE
- Update CODEMAP.md after backend simplification

## [5.0.1] - 2026-03-25

### Fixed

- Fix CI smoke tests for v5.0 simplification: remove `daxpy` import from C extension (moved to numpy-only), handle missing vendor LAPACK gracefully in eigh smoke test

## [5.0.0] - 2026-03-25

### Changed

- **BREAKING**: Remove JAX backend — NumPy+C is now the only compute path
- Strip own-BLAS/LAPACK C implementations (dgemm, dsyrk, dsytrd, dstedc, dormtr); vendor-only dispatch
- Archive JAX runners, tests, and scripts to `legacy/`
- Simplify jlinalg to vendor-BLAS-only dispatch (ILP64 MKL/OpenBLAS/Accelerate → NumPy fallback)
- Add clang-format and cppcheck pre-commit hooks for C extensions
- Add SeededETA progress bars with model-predicted initial ETAs
- Net -21,900 lines removed

## [4.6.3] - 2026-03-24

### Changed

- Raise maximum covariate limit from 20 to 100 in C extension (MAX_N_CVT)

## [4.6.2] - 2026-03-23

### Changed

- Eigendecomp log now shows driver name (DSYEVD-inplace/DSYEVD/DSYEVR) instead
  of generic `jlinalg.eigh`, explains why that driver was chosen (e.g. "kinship
  in memory, overwriting in place"), and lists the relevant alternative with its
  memory cost (e.g. "DSYEVR fallback=126.3GB")

## [4.6.1] - 2026-03-23

### Fixed

- Prefer clang over GCC when linking libiomp5 — GCC's GOMP compatibility shim
  triggers assertion failures (`kmp_runtime.cpp` Error #13) after MKL LAPACK
  operations (e.g. DSYEVR). Clang natively generates `kmp_*` calls that
  libiomp5 handles correctly.
- Simplify clang OpenMP detection to avoid `omp.h` dependency and `-x none`
  parsing issues with libiomp5.so paths
- Add `JLINALG_NO_VENDOR_LAPACK` env var to skip MKL dsyevd/dsyevr in eigh,
  falling back to jlinalg-own LAPACK
- Respect `JLINALG_NO_VENDOR_LAPACK` in eigendecomp driver selection
- Replace OpenMP with pthreads in `compute_snp_stats_chunk` to avoid
  MKL/libiomp5 conflict — SNP stats is memory-bandwidth-bound, not compute-bound
- Auto-recompile jlinalg C extension on import failure (stale `.so`)

### Changed

- Centralize jlinalg thread control: new `jlinalg_threads()` context manager
  with RLock for thread-safe `set_n_threads()` scoping (replaces ad-hoc
  `blas_threads()` calls for jlinalg rotation in runners)
- Centralize C extension OpenMP detection: `get_c_extension_capabilities()`
  returns `(available, has_openmp)` tuple; `get_c_extension_thread_count()`
  consolidates thread sizing logic
- Chunk `compute_snp_stats()` in 10k-SNP slices to avoid full contiguous
  copy of large genotype matrices
- `detect_openmp_flags()` returns `cc_override` as third element when
  switching to clang for libiomp5 compatibility
- Fix pipeline thread logging for serial (no-OpenMP) C extension builds

## [4.6.0] - 2026-03-23

### Added

- `JAMMA_NO_OPENMP=1` environment variable to compile C extensions without
  OpenMP — completely avoids dual OpenMP runtime SIGABRT on Databricks where
  both Intel OMP (MKL) and GNU OMP (scipy) are pre-loaded by the kernel before
  any user code runs. Single-threaded C extensions are still much faster than
  pure-Python fallback.

## [4.5.3] - 2026-03-23

### Fixed

- Move `KMP_DUPLICATE_LIB_OK` to `jamma/__init__.py` (earliest import point) —
  on Databricks, `mkl._mklinit` and scipy are loaded by the kernel before
  `jlinalg/__init__.py` runs, so the v4.5.2 fix was too late

### Changed

- Consolidate OpenMP detection into `core.openmp_detect` — eliminates 3-way
  duplication across `_compile_accel.py`, `_compile_jlinalg.py`, and
  `hatch_build.py` (hatch_build.py keeps its own copy with a sync comment)

## [4.5.2] - 2026-03-23

### Fixed

- Set `KMP_DUPLICATE_LIB_OK=TRUE` before C extension import to prevent dual
  OpenMP runtime SIGABRT on Databricks — scipy (pre-loaded by kernel) brings
  libgomp while jlinalg/`_lmm_accel` link against MKL's libiomp5

## [4.5.1] - 2026-03-23

### Fixed

- Two-step compile+link for `_lmm_accel` to prevent dual OpenMP runtime SIGABRT
  on Linux with MKL numpy — GCC's `-fopenmp` implicitly links libgomp alongside
  libiomp5, causing `kmp_runtime.cpp` assertion failure

## [4.5.0] - 2026-03-23

### Added

- Split general Score/LRT C entry points (`compute_score_split_general_c`,
  `compute_lrt_split_general_c`) — accept SoA data directly, eliminating
  `reconstruct_uab_from_soa` for n_cvt>1 (~75 GB saved at n_cvt=2/100k samples)
- `out=` buffer reuse for general n_cvt in `batch_compute_uab_varying_soa_numpy` —
  zero per-chunk allocation for varying SoA across all covariate counts
- `logdet_from_row0` helper — deduplicates 3 inline identity Pab prepass blocks
- Fused general mode-4 dispatch for n_cvt≥2 — all 8 output arrays (Wald + Score +
  LRT) computed in a single workspace pass

### Fixed

- Mode-4 fused general availability guard now checks `_C_MODE4_FUSED_GENERAL_AVAILABLE`
  (previously used Wald-only flag)
- `out=` buffer validates dtype (float64) and C-contiguity
- OpenMP compile/link flag split to prevent dual-runtime SIGABRT (libgomp + libiomp5)
- Chunk-size accounting for n_cvt>1 Score/LRT reflects split C dispatch (no Uab
  reconstruction overhead)

## [4.4.2] - 2026-03-23

### Fixed

- Use actual inplace DSYEVD memory requirement for DSYEVR fallback decision instead
  of always using the non-inplace peak estimate
- Guard `out=` buffer allocation behind `n_cvt==1` in batch and streaming NumPy
  runners — `batch_compute_uab_varying_soa_numpy` only supports it for single-covariate
- Improve `dispatch_soa_split` error message for unreachable mode-4 path
- Simplify no-DSYEVR fallback branch — no longer silently downgrades inplace to
  conservative estimate

## [4.4.1] - 2026-03-22

### Changed

- Updated benchmark table with best NumPy+C numbers — Wald 879ms (12.5x vs GEMMA), All 16.0x

## [4.4.0] - 2026-03-21

### Added

- Early sample filtering via `valid_indices` — missing-phenotype samples are
  excluded before kinship accumulation rather than post-hoc, avoiding full n×n
  matrix materialisation (kinship streaming, LOCO NumPy, LOCO JAX, PipelineRunner)
- Input validation (`_validate_valid_indices`) for LOCO NumPy kinship streamer
- Filtered sample count in LOCO log messages for both NumPy and JAX backends

### Removed

- Secular equation solver and LOCO streaming modes (`S_CHR`, `X_C`,
  `X_C_SEQUENTIAL`) — superseded by streaming LOCO with better memory
  characteristics
- `--secular` CLI flag and `use_secular_update` config option
- `loco_eigen_update.py` (1090 lines) and associated tests (~2200 lines)

### Fixed

- Replace `assert` with `raise ValueError` for kinship shape validation in
  pipeline (assert stripped by `python -O`)
- Remove stale documentation references to deleted secular update feature

## [4.3.1] - 2026-03-21

### Added

- Pipeline machinery for NumPy streaming runner — overlaps DGEMM rotation of
  chunk N+1 with C extension compute of chunk N via ThreadPoolExecutor
  double-buffering, with adaptive core splitting and memory-aware chunk sizing

### Changed

- Swap utg_t layout to (n_snps, n_samples) for direct DGEMM TRANSA — eliminates
  post-rotation transpose in batch and streaming NumPy runners
- Add GEMMA Accelerate to backend comparison benchmark

### Fixed

- Avoid O(n²) eigenvector copy in streaming chunk loop
- Rename unused loop variable to satisfy linter

## [4.3.0] - 2026-03-21

### Added

- Fused general C kernels for arbitrary n_cvt Wald test — eliminates Python-level
  Uab reconstruction loop for multi-covariate models (n_cvt ≥ 2)
- Availability flags (`_C_FUSED_GENERAL_AVAILABLE`, `_C_MODE4_FUSED_GENERAL_AVAILABLE`)
  with workspace creation and dispatch functions
- Runner integration test for n_cvt=2 end-to-end fused vs non-fused validation

### Changed

- Batch and streaming runners auto-dispatch fused general path when n_cvt ≥ 2
  and C extension is available
- Updated PERFORMANCE.md and time estimates to v4.2.0 benchmarks (2h 29m at 125k)
- Removed DSYEVR time multiplier — empirically comparable to DSYEVD at scale

### Fixed

- Input validation hardening — bounds checks on table indices, var columns,
  n_snps in C kernels
- Mode-4 fused general disabled at dispatch level due to NaN lambda_mle bug

## [4.2.1] - 2026-03-20

### Fixed

- Link Intel OpenMP by full path in hatch_build.py — numpy bundles versioned
  names like `libiomp5-2f035e84.so` with no unversioned symlink, so `-liomp5`
  fails at link time

## [4.2.0] - 2026-03-20

### Changed

- Fused Uab compute — reduces peak memory for NumPy batch and streaming runners
  by computing Uab in a single pass instead of separate U.T @ W and U.T @ y steps
- Complete analytical dev2 for all n_cvt values (previously only n_cvt=1)
- Deduplicate cleanup_jax_caches and fix per-chromosome cache clearing in LOCO
- Extract shared PASS 1 + setup into _loco_chr_common for LOCO runners
- Extract try/finally bodies to _impl() helpers in JAX runners
- Remove private import aliasing in loco.py

### Fixed

- Decouple DSYEVR/DSYEVD attribute checks in pre-flight memory estimate
- Relax eigh inplace eigenvalue tolerance from 1e-12 to 5e-12 for CI stability
- Relax pve_se assertion for synthetic data with no signal
- Fix flaky memory test and add mode-4 threading parity test

### Removed

- Lazy eigendecomposition (phases 89, 89.1) — dstedc workspace (3N²) exceeds
  DSYEVR memory at scale, making the lazy path unviable for 100k+ samples

## [4.1.0] - 2026-03-19

### Changed

- `jlinalg.eigh` gains `inplace` keyword — when `inplace=True`, eigenvectors are
  written directly into the input K buffer, avoiding one N×N allocation (~125 GB
  savings at 125k samples). Requires vendor DSYEVD (ILP64 BLAS).
- `eigendecompose_kinship` automatically uses `inplace=True` when vendor DSYEVD is
  available and DSYEVD fits in memory
- Memory estimator (`check_memory_before_run`) accounts for in-place path, producing
  tighter estimates when vendor DSYEVD is available
- Add `_dsyevd_inplace_peak_gb` memory estimator for the in-place eigendecomp path

### Fixed

- Remove unused `null_inv_ww` variable in `compute_score_split_c` (_lmm_accel.c)
- Document FP tolerance rationale in streaming NumPy test

## [4.0.3] - 2026-03-18

### Fixed

- _lmm_accel compile summary now correctly reports "single-threaded" when
  OpenMP fallback was used, instead of always reporting "OpenMP"

## [4.0.2] - 2026-03-18

### Changed

- C extension compile scripts now default to quiet output — only errors and a
  one-line summary are printed. Pass `verbose=True` for full per-command detail.

## [4.0.1] - 2026-03-18

### Fixed

- Link Intel OpenMP (libiomp5) by full path in C extension compile scripts — numpy
  bundles versioned names like `libiomp5-2f035e84.so` with no unversioned symlink,
  so `-liomp5` fails at link time
- Add OpenMP link fallback in jlinalg compile — retries without OpenMP flags if
  linking fails, producing a single-threaded build instead of a hard error

### Changed

- **Eigendecomposition now uses jlinalg.eigh** — replaced the legacy `_eigen_accel`
  C extension and `numpy._umath_linalg.eigh_lo` gufunc cascade with unified
  `jlinalg.eigh`, which dispatches to vendor DSYEVD/DSYEVR or the jlinalg D&C
  pipeline depending on available BLAS backends
- Add DSYEVR vendor dispatch to jlinalg C layer — memory-pressure fallback with
  O(N) workspace vs O(N²) for DSYEVD, ILP64-only
- Wire `jlinalg.dsyrk` into kinship and `jlinalg.dgemm` into prepare
- Expose `jlinalg.blas_has_dsyevr` capability flag
- `jlinalg.eigh` now raises `numpy.linalg.LinAlgError` (not `RuntimeError`) on
  convergence failure
- Memory estimator simplified: removed `_inplace_eigen_available()` check since
  jlinalg.eigh always allocates separate eigenvectors
- DSYEVR availability check in `check_memory_before_run()` now queries
  `jlinalg.blas_has_dsyevr` instead of importing from `eigen.py`
- **Rename `jblas` package to `jlinalg`** — the package now covers BLAS, LAPACK,
  and LAPACKE dispatch (not just BLAS), so `jlinalg` ("JAMMA linear algebra")
  better reflects its scope. All imports, C function prefixes (`jlinalg_*`),
  macros (`JLINALG_*`), and file paths updated.

### Removed

- Delete legacy `_eigen_accel` C extension and `_secular_accel` C extension
  source + compile script (`_secular_accel.c`, `_compile_secular.py`);
  LOCO secular path now always uses Python fallback
- Remove `INPLACE_EIGEN_AVAILABLE` flag and `_eigh_inplace()` gufunc path
- Remove `_DSYEVR_AVAILABLE`, `_try_import_dsyevr()`, `_lazy_init_dsyevr()`,
  `_select_eigen_driver()`, `_eigh_dsyevr()` from `eigen.py`
- Remove `_inplace_eigen_available()` from `memory.py`

## [4.0.0] - 2026-03-18

### Added

- **NumPy streaming runner** — disk-streaming LMM association using the C
  extension, matching the JAX streaming runner's two-pass architecture
  (float32 stats pass then float64 compute pass) with incremental I/O
- Wire numpy-streaming into pipeline, CLI (`--backend numpy-streaming`),
  backend selection, and benchmark suite

### Fixed

- Thread-safe P_yy warning deduplication — replace global `bool` flags with
  `threading.local()` in `likelihood.py` and `likelihood_numpy.py`
- Add `get_last_run_timing()` accessor to `runner_jax.py` matching the
  pattern in streaming runners; pipeline uses accessor instead of directly
  importing the mutable module-level dict
- Inline `_calc_pab_general` into `calc_pab`, removing unnecessary
  indirection layer
- Use keyword arguments for `AccelImport` NamedTuple construction to prevent
  positional field mismatch in the 17-field type
- Narrow `_check_hwe_support` to numpy-batch only (was incorrectly guarding
  all numpy paths)

### Changed

- Exclude `tier3` marker from default pytest addopts — the 22-minute
  `test_secular_speedup_correctness_at_scale` was running on every invocation
- Mark eigendecomp symmetry check tests and LOCO eigen cache integration
  tests as `slow` (15–35s each)

## [3.5.1] - 2026-03-12

### Fixed

- Ship `_secular_accel.c` C extension in wheel — was missing from
  `hatch_build.py`, causing Databricks to fall back to Python rank-1 update
  which allocates n×n dense matrices (58 GB at n=85k) and segfaults
- Guard Python fallback rank-1 updates with `MemoryError` at n > 10k to fail
  fast with actionable message instead of silent segfault

## [3.5.0] - 2026-03-12

### Added

- Benchmark telemetry module (`core/telemetry.py`) — appends structured JSONL
  run records to `~/.jamma/benchmarks.jsonl` with `JAMMA_NO_TELEMETRY` opt-out
- `n_cvt`-aware backend selection — `select_execution_mode` accounts for
  covariate count in memory estimates and falls through to JAX when C general
  extension is unavailable for `n_cvt > 1`
- Telemetry emission from `PipelineRunner.run()` via `_emit_telemetry()` helper
  (both LOCO and standard paths)

## [3.4.1] - 2026-03-11

### Changed

- Make `deflated` a required parameter in blocked Cauchy multiply functions,
  preventing silent fallback to approximate 0/0 handling
- Remove redundant `n` parameter from `_check_and_reorthogonalize` helper
- Replace O(n) `argmin` with O(log n) `searchsorted` in deflated column detection
- Lazy `argsort` — check `np.diff >= 0` before sorting eigenvalues
- Deduplicate eigen write block and `batch_chr_set` in LOCO orchestrator

## [3.4.0] - 2026-03-11

### Added

- LOCO secular equation solver — O(n^2 * r_eff) eigenvalue perturbation path
  replacing O(n^3) `np.linalg.eigh` for leave-one-chromosome-out eigendecomposition.
  Enabled via `--secular` CLI flag or `PipelineConfig(secular=True)`
- C extension (`_secular_accel.c`) implementing LAPACK DLAED4-based rank-1
  eigenvalue solver with negative-rho handling via negation/reversal identity
- Delta-path eigenvector recomputation eliminating 55 GB `Q = np.eye(n)` allocation
  at n=83k. Two-pass algorithm with blocked Cauchy multiply and pre-allocated buffers
- `LocoStreamingMode` enum and `SequentialLocoResult` NamedTuple for type-safe
  streaming mode dispatch in LOCO pipeline
- `yield_x_c_sequential` streaming mode for one-chromosome-at-a-time secular processing
- Orthogonality monitoring (`check_orthogonality`) and `reorth_interval` parameter
  in secular solver for numerical stability tracking
- `bench_secular.py` benchmark script for secular solver performance profiling

### Changed

- `SecularImport` NamedTuple + named constants for secular solver clarity
- Extract `_cauchy_block` helper, deduplicating 6 call sites in eigenvector reconstruction
- Deflation guard in C eigenvector path, NaN check in delta forward pass

## [3.3.2] - 2026-03-10

### Fixed

- LOCO `save_kinship` log message showed `.txt` path but `write_kinship_matrix()`
  actually writes `.npy` (binary default since v2.11). Now logs the actual path written.

## [3.3.1] - 2026-03-10

### Fixed

- LOCO multi-pass batch sizing now reserves eigendecomposition workspace memory.
  Previously the batch sizer allocated too many S_chr matrices per pass, leaving
  insufficient memory for DSYEVR/DSYEVD when eigendecomp runs while the generator
  is suspended with remaining S_chr matrices alive (OOM on 85k+ samples, 40 chromosomes)
- Fix `single_pass_gb` formula in JAX LOCO path — was `matrix_gb * (1 + n_chr)`,
  now `(2 + n_chr)` to account for K_loco_buf
- Fix `min_required_gb` to include eigendecomp workspace and K_loco_buf in both
  JAX and NumPy paths

## [3.3.0] - 2026-03-10

### Added

- NaN diagnostic accumulation in streaming runner — tracks per-key NaN counts
  across chunks and logs warnings with actionable advice (degenerate genotypes,
  kinship quality)

### Changed

- Extract `_guarded_compute` helper to DRY up 8 duplicated try/except error-
  wrapping blocks in NumPy runner with operation-specific labels for diagnosis
- Add `dtype.kind` guard on NaN check to prevent diagnostic from crashing on
  non-float arrays
- LMM All (`-lmm 4`) NumPy+C: 5.5s → 1.4s on mouse_hs1940 (3.6x faster,
  14.3x vs GEMMA) — removing per-SNP exception frame overhead from hot loop

## [3.2.0] - 2026-03-09

### Added

- LOCO per-chromosome eigen cache (`--eigen-dir`) — saves eigendecomposition
  results per chromosome and reloads them on subsequent runs, skipping both
  kinship computation and eigendecomposition entirely
- `_find_loco_eigen_cache()` helper validates cache completeness before use;
  partial or missing caches fall back to full compute transparently
- `-eigen` flag now works with `-lmm -loco` to write per-chromosome eigen files
  (previously only supported with `-gk`)
- `write_eigen`, `eigen_dir`, `eigen_prefix` parameters on `run_lmm_loco()`
- Dimension validation on cached eigen load with chromosome-contextual errors
- `-d`/`-u` (pre-computed global eigen) now blocked with `-loco` with clear
  error message directing users to `--eigen-dir`

## [3.1.0] - 2026-03-07

### Added

- PVE standard error (`pve_se`) computed via delta method from REML second
  derivative — available in `LmmRunResult`, `LocoResult`, `PipelineResult`,
  and `GWASResult`
- `LocoResult` dataclass replaces raw tuple return from `run_lmm_loco()`,
  with named fields: `associations`, `n_tested`, `pve`, `pve_se`
- `finite_difference_dev2()` — numerical REML second derivative via central
  finite differences; used for `pve_se` computation for all covariate counts
- `reml_log_likelihood_dev2()` — partial analytical REML second derivative
  (intercept-only); delegates to `finite_difference_dev2` for n_cvt > 1
- `calc_ppab()` and `calc_pppab()` — second/third-order projected Pab
  recursions (ports of GEMMA's `CalcPPab`/`CalcPPPab`)
- Finite-difference tests validate second derivative for n_cvt=1,2,3,4
- `jax.clear_caches()` now runs in `finally` blocks across all runners,
  with defensive `try/except` to avoid masking original exceptions

### Fixed

- `reml_log_likelihood_dev2()` was missing the d²(logdet_hiw)/dλ² term,
  producing incorrect REML curvature for multi-covariate models (n_cvt > 1).
  `compute_and_log_pve()` now uses `finite_difference_dev2()` for all n_cvt

### Breaking

- `run_lmm_loco()` returns `LocoResult` dataclass instead of
  `tuple[list, int, float | None, float | None]`

## [3.0.1] - 2026-03-06

### Fixed

- Streaming memory estimates now distinguish disk chunk size (raw genotype buffer)
  from JAX sub-chunk size (rotation/Uab/grid buffers), producing accurate LMM
  phase estimates after per-subchunk flush

## [3.0.0] - 2026-03-06

### Breaking

- `LmmRunResult` no longer supports list-like access (`len()`, iteration,
  indexing, `bool()`). Use `.associations` explicitly:

  ```python
  # Before (2.x)
  results = run_lmm_association_numpy(...)
  for r in results: ...

  # After (3.0)
  run_result = run_lmm_association_numpy(...)
  for r in run_result.associations: ...
  ```

### Added

- `_chunk_result_to_numpy()` — transfers JAX sub-chunk results to host
  immediately instead of accumulating on device until disk chunk completes
- PVE capture in LOCO is now robust to filtered first chromosomes — falls back
  to the next chromosome with passing SNPs
- Warning logged when PVE cannot be computed (all chromosomes fully filtered)
- Regression tests for per-sub-chunk flushing (disk-write and in-memory paths)
- PVE cross-backend parity assertions in LOCO tests

### Changed

- Streaming and LOCO runners flush each JAX sub-chunk to host/disk immediately,
  reducing peak device memory from O(disk_chunk) to O(jax_chunk)

### Removed

- Dead code: `strip_and_append`, `_concat_jax_accumulators`,
  `_init_accumulators`

## [2.12.0] - 2026-03-06

### Added

- Invariant/varying Uab column split for general n_cvt — correctly classifies
  columns as lambda-invariant or lambda-varying based on covariate structure
- Consolidated pipeline startup logging into a single banner line

### Fixed

- JAX batch LMM memory estimate used max chunk size instead of actual chunk size,
  causing unnecessary chunking on smaller datasets
- JAX batch memory safety factor reduced from 1.5x to 1.25x to avoid over-conservative
  chunk splitting
- Pipeline banner logging hardened against missing backend diagnostics
- Technical RSS labels replaced with plain English in log messages

### Changed

- Extracted `_prepare_general_split_inputs` to deduplicate column setup across
  Uab split paths
- Simplified banner formatting code

### Documentation

- Added cross-references between LOCO test files
- Clarified `gwas()` as recommended API, Intel CPU optimization, platform BLAS details

## [2.11.2] - 2026-03-05

### Fixed

- Test expected text `.cXX.txt` kinship path but default output is now binary `.cXX.npy`

### Changed

- Consolidated pipeline startup logging into a single banner line showing runner, BLAS backend, eigen driver, C extension status, and thread count
- Updated project logo

## [2.11.1] - 2026-03-05

### Fixed

- Multi-phenotype runs crashed with eigenpair dimension mismatch when phenotype
  missingness differed across columns — now NaN-stamps samples outside the shared
  valid_mask intersection so runners compute a consistent mask
- `JAMMA_BACKEND` environment variable was ignored when `backend="auto"` — now
  resolved before auto-selection logic
- Backend logging falsely attributed selection to `JAMMA_BACKEND` when env var was
  set but not actually honored; removed misleading "JAX not installed" message for
  memory/C-extension-based NumPy selection
- NumPy multi-phenotype runs reloaded full genotype matrix per phenotype — now
  pre-loads PLINK data once
- Windows + JAX docs contradiction between README and User Guide
- "Full test suite" claim in PERFORMANCE.md now notes default marker exclusions

### Changed

- Extracted `compute_valid_mask()` to `prepare_common.py` — single source of truth
  for valid-sample mask logic (was duplicated in pipeline, prepare_common, loco)
- Added `get_last_run_timing()` accessor for thread-safe timing snapshot

## [2.11.0] - 2026-03-05

### Added

- Binary `.npy` as default output format for kinship matrices and eigendecomposition
  files — 10-100x faster I/O at scale. Use `--legacy-text` for GEMMA-compatible text format
- Multi-phenotype support: `-n "1 2 3"` or `-n "1,2,3"` processes multiple phenotype
  columns with a single eigendecomposition, saving hours at scale
- Shared `npy_cache` module for `.npy` sibling validation logic

### Changed

- Kinship output file extension changed from `.cXX.txt` to `.cXX.npy` by default
- Eigen output files changed from `.eigenD.txt`/`.eigenU.txt` to `.eigenD.npy`/`.eigenU.npy`
  by default

## [2.10.1] - 2026-03-03

### Fixed

- Golden section optimizer returned inconsistent (lambda, logl) pair — lambda
  at midpoint `(a+b)/2` but logl as `max(fc, fd)` from different points c and d.
  Now evaluates logl at the midpoint, matching the JAX path. This eliminates
  cross-backend p_lrt divergence (4.5e-4 → 1.05e-10 on gemma_synthetic)
- `compare_assoc_results` LRT mode used `pvalue_rtol` (1e-4) instead of
  `p_lrt_rtol` (5e-3) for p_lrt comparison
- C vs Python parity test compared C extension (generic golden section) against
  Python split-Uab optimizer — now calls generic optimizer directly
- `check_memory_before_run` passed defaults to `_compute_chunk_size` instead of
  `n_samples` and `pipeline_buffers=2`, causing overestimated memory
- `_compute_chunk_size_numpy` lacked `pipeline_buffers` type/range validation
- Exposed rotation time metric could exceed total rotation time due to
  GC/scheduling jitter — now capped at `rot_dur`

### Added

- MemoryError passthrough tests for both JAX batch and streaming runners
- `pipeline_buffers` TypeError tests (float/str/None) for all chunk sizers
  and memory estimators

## [2.10.0] - 2026-03-03

### Added

- Rotation-compute overlap pipelining — both JAX batch and streaming runners
  overlap BLAS rotation (U.T @ G) with XLA compute using a `ThreadPoolExecutor`
  background thread, achieving ~15% wall-time reduction on mouse_hs1940
- `pipeline_buffers` parameter for `_compute_chunk_size` and streaming memory
  estimators to account for double-buffered UtG arrays during overlap
- Input validation for `pipeline_buffers` (type check, >= 1 guard)
- `MemoryError` passthrough in both runners to avoid wrapping OOM as RuntimeError
- Background rotation failure propagation tests with exception chaining
- Multi-file-chunk `prev_compute_end` handoff test for streaming runner
- Rotation overlap effectiveness tests (timing invariants)

### Fixed

- Streaming runner `ThreadPoolExecutor` scope hoisted to span BED file-chunk
  boundaries, fixing `prev_compute_end` timing handoff across chunks
- Memory estimators in `check_memory_before_run` and streaming runner now pass
  `pipeline_buffers=2` for accurate double-buffer accounting

## [2.9.6] - 2026-03-03

### Added

- Device-memory-aware JAX chunk sizing — auto-scales to GPU/TPU memory budget
  with psutil fallback for CPU
- Filtered reads and threaded prefetch iterator for streaming runner —
  `snp_indices` column-selection in PLINK reader skips unneeded genotype columns
- Multi-pass chromosome batching for NumPy LOCO kinship — streams BED in
  multiple passes when all per-chromosome matrices don't fit in memory
- `SnpStatsCache` — caches global SNP statistics from kinship pass, eliminating
  redundant per-chromosome BED re-reads in the association phase
- Valid-indices threading — propagates phenotype-valid sample indices into
  kinship streaming so K_loco is built at n_valid × n_valid directly
- In-place K_loco buffer reused across chromosomes (caller must eigendecompose
  before advancing)
- `JAMMA_LOCO_WORKERS` env var for LOCO parallel execution control
- Imputation guard raises on >50% missing rate before centering
- 500+ lines of new tests: chunk tuning, split-Uab modes, LOCO aliasing,
  filtered reads, streaming edge cases, multi-pass equivalence, valid-sample
  subsetting

### Changed

- Split-Uab for all LMM modes — LRT/Score/All reconstruct full Uab from split
  SoA components with correct 9-col peak memory accounting
- Adaptive core split (`compute_pipeline_core_split`) replaces fixed 75/25 split
  with min-2 / fallback logic
- BLAS controllability detection gracefully falls back when Accelerate (macOS) is
  the BLAS backend
- DRY refactors in plink.py and runner_numpy.py, structured error handling in
  chunk.py
- Documentation: quote `'jamma[jax]'` in shell contexts (zsh glob fix), remove
  misleading GPU Support section

### Fixed

- K_loco aliasing bug — copy buffer before yielding to prevent all chromosomes
  sharing a single array
- SnpStatsCache stores `n_samples` (all-sample population denominator) —
  prevents inflated miss_rates when n_valid < n_samples
- `_s_full_accumulated` assert prevents S_full double-counting in LOCO
- Strict snp_indices validation (ascending + bounds), removes tail-chunk NaN
  padding

## [2.9.5] - 2026-03-02

### Added

- AVX2-optimized wheel build job in CI — builds with `-march=x86-64-v3 -mavx2`,
  verifies AVX2 instructions via `objdump`, attaches to GitHub releases
- `aligned_alloc` for C extension workspace arrays (32-byte AVX2 alignment)
- ABI mismatch detection — stale `.so` fallback logged via `loguru.warning`

### Changed

- Fused Wald computation into golden section optimizer — eliminates redundant
  `n_samples` pass to recompute `hi_eval` at `lambda_opt` by reusing the buffer
  from the final REML evaluation
- C extension build: `CFLAGS` passthrough, `-funroll-loops`,
  `-fno-finite-math-only` safety, C11 standard, `schedule(static)` for uniform
  SNP cost
- C vs Python parity test uses well-conditioned synthetic data (proper w×x
  cross-products) with calibrated tolerances from measured FP differences

### Fixed

- Degenerate SNP hardening — negative P_YY guard (Schur complement), early-return
  when every grid point is NaN (`REML_SENTINEL` pattern), explicit `is_valid`
  return from `wald_from_pab`, p-value clamping to [0,1]
- C extension validity checks hardened with input shape and scalar parameter
  validation
- README: removed false GPU acceleration claims, fixed architecture diagram

## [2.9.4] - 2026-03-02

### Changed

- `impute_and_center()` operates in-place on writable NumPy arrays, eliminating an
  O(N×M) copy during kinship computation (KIN-03)
- `impute_center_and_standardize()` uses `np.einsum('ij,ij->j')` for variance
  computation instead of materializing an O(N×M) `X**2` intermediate (KIN-06)
- `compute_loco_kinship()` rewritten in pure NumPy — no longer initializes JAX
  during in-memory LOCO kinship computation (KIN-01, KIN-04)
- `_ensure_float64()` skips copy when input is already float64 (KIN-02)
- Per-chromosome `block_until_ready()` calls added to streaming LOCO accumulation
  to prevent unbounded JAX async dispatch (KIN-05)
- `_compute_chunk_size()` simplified: removed vestigial `n_samples`/`bytes_per_element`
  parameters, uses `MAX_SAFE_CHUNK` cap directly

### Fixed

- Streaming LOCO `S_chr` matrices were not synchronized before subtraction, which
  could produce stale results under heavy JAX async dispatch

## [2.9.3] - 2026-03-01

### Added

- Runtime LAPACK discovery via dlopen — `_eigen_accel` no longer has link-time LAPACK
  dependency, making compiled wheels portable across numpy builds (OpenBLAS, MKL, Accelerate)
- `scipy_dsyevr_64_` symbol resolution for PyPI numpy wheels (scipy-openblas64 uses
  `scipy_` prefix on all LAPACK symbols)
- Intel OpenMP (`libiomp5`) detection for `_lmm_accel` — avoids libgomp/libiomp5
  dual-runtime conflict on MKL systems
- `EIGEN_ACCEL_DEBUG=1` environment variable for LAPACK discovery diagnostics
- `IS_ILP64` constant exported from `_eigen_accel` module

### Changed

- LAPACK discovery tries symbols in priority order: `dsyevr_64_` → `scipy_dsyevr_64_`
  → `dsyevr64_` (ILP64), then `dsyevr_` (LP64)
- `_eigen_accel` ABI version bumped to 2 (dlopen rewrite)
- Linux wheels no longer need system LAPACK — dlopen resolves from numpy's bundled BLAS

### Fixed

- Linux CI: `_eigen_accel` DSYEVR resolution failed because PyPI numpy bundles
  scipy-openblas64 with `scipy_` prefixed symbols
- Module init: replaced `PyRun_String` with C API calls — `__builtins__` is unavailable
  in globals dict during module init, causing silent import failures
- Linux dlopen: uses `/proc/self/maps` scan after forcing numpy BLAS load to find
  libraries opened with `RTLD_LOCAL` (invisible to `RTLD_DEFAULT`)

## [2.9.2] - 2026-03-01

### Added

- DSYEVR C extension for eigendecomposition — O(N) workspace vs O(N²) for DSYEVD,
  saving ~250GB at 125k samples; auto-compiled on first use with lazy recompilation
- LAPACK linkage for Linux wheels: auto-detects numpy's bundled OpenBLAS in numpy.libs/
  for C extension compilation (both hatch_build.py and post-install_compile_eigen.py)
- Negative n_samples validation in memory estimation functions
- ABI mismatch test for DSYEVR import probe

### Changed

- Eigendecomposition now prefers DSYEVD (1.2–1.5x faster) by default, falling back
  to DSYEVR only when DSYEVD workspace exceeds available memory
- Memory estimates default to DSYEVD (conservative); actual peak is lower if DSYEVR
  is triggered
- DSYEVR auto-recompilation deferred from module import to first eigendecomp call
  (avoids subprocess/compiler side effects during import)
- DSYEVR workspace query uses ceil() to prevent off-by-one undersized allocation
- DSYEVR type stub accepts lowercase UPLO values ('l', 'u')
- Memory comment corrected: DSYEVR saves ~250GB (not ~232GB) at 125k samples

### Fixed

- DSYEVR fallback when neither driver fits: now uses DSYEVR (smaller peak) with
  OOM warning instead of silently falling through to DSYEVD
- Matrix reader: MemoryError re-raised directly instead of wrapping in RuntimeError
- Matrix reader: temp dir fallback includes OS error message in warning
- Class-level import in TestLambdaBoundaryDiagnostics moved to method level
  (prevents import error when results module unavailable)

## [2.9.1] - 2026-03-01

### Fixed

- Platform-tagged wheels: set `pure_python=False` and `infer_tag=True` in hatch build
  hook so cibuildwheel produces platform-specific wheels (e.g. `cp311-cp311-manylinux_2_28_x86_64`)
  instead of `py3-none-any`
- Upgrade cibuildwheel v2.22.0 to v3.3.1 (fixes stale manylinux2014 image reference)
- Switch `CIBW_BEFORE_BUILD_LINUX` from `yum` to `dnf` (manylinux_2_28 uses AlmaLinux 8)
- macOS wheels compile C extension single-threaded (no OpenMP) to avoid delocate
  `MACOSX_DEPLOYMENT_TARGET` conflict with Homebrew libomp
- Replace inline `python -c` test with standalone smoke test script to avoid shell
  escaping and indentation issues in cibuildwheel containers

## [2.9.0] - 2026-03-01

### Added

- **C extension for NumPy LMM runner** (`_lmm_accel.c`) — OpenMP-parallelized Wald test
  replaces Python loop over SNPs. Includes workspace API (`create_workspace` /
  `compute_wald_stats_workspace`) that pre-allocates all per-thread buffers once per chunk,
  and SoA-native Uab generation with invariant precompute to eliminate redundant work
- **Split-Uab C extension** — SoA (struct-of-arrays) layout for split Uab computation
  with internal Iab precompute, avoiding Python-side Iab construction entirely
- **Parallel matrix text I/O** (`matrix_reader.py`) — multi-worker `.eigenD.txt` /
  `.eigenU.txt` reader with chunk-boundary scanning and `np.loadtxt` per chunk
- **Eigen I/O `.npy` sidecar cache** — binary cache with mtime-based invalidation for
  eigenvalue/eigenvector files (3s warm read vs 4min cold text parse at 50k samples)
- **CI wheel build workflow** (`build-wheels.yml`) — cibuildwheel for manylinux x86_64
  and macOS arm64 wheels with OpenMP support
- Static OpenMP schedule for deterministic thread assignment across chunks

### Changed

- NumPy runner auto-detects C extension availability and dispatches to accelerated path
  when `n_cvt=1`, falling back to pure Python otherwise
- Memory estimator (`estimate_streaming_memory`) accounts for C extension workspace
  allocation when the accelerated path is active
- BLAS thread coordination: rotation threads and compute threads are balanced to prevent
  oversubscription during OpenMP regions
- Publish workflow includes wheel artifacts from build-wheels CI

## [2.8.3] - 2026-02-27

### Changed

- Local pytest defaults to `-n 3`, `--no-cov`, and skips `slow`/`tier2` tests to reduce memory pressure on dev machines
- CI overrides `addopts` with `-o 'addopts='` to run full suite with coverage independently of local config

## [2.8.2] - 2026-02-27

### Fixed

- **Critical: NaN propagation in golden section optimizer** — if the first grid point
  returned NaN (degenerate kinship), the scalar optimizer stayed stuck at NaN forever,
  silently producing NaN results for the entire GWAS run. Now initializes `best_val=inf`
  and skips NaN grid points.
- **Critical: `argmax` on NaN-containing grids** — JAX/NumPy `argmax` could select NaN
  entries as "best", causing the golden section to refine around a garbage bracket.
  NaN entries are now replaced with `-inf` before `argmax` in both batch paths.
- **Negative eigenvalues now zeroed in `eigendecompose_kinship`** — previously only warned.
  Negative eigenvalues above the threshold (e.g. -1e-5) survived into likelihood computation
  where `np.abs(v_temp)` silently masked incorrect logdet values.
- Missing `kinship is None` guard in JAX runner (NumPy runner already had it)
- Missing `lmm_mode` validation in `_compute_lmm_chunk` (JAX compute dispatch)
- `block_chunk_result` could `AttributeError` on `None` values for unexpected modes

### Changed

- Batch `_guard_P_yy` now logs a warning when negative P_yy values are detected
  (previously silent, unlike the scalar `_clamp_p_yy` path)
- Scalar Pab recursion now logs debug message for degenerate `ps_ww=0` entries
- Runners now emit per-key NaN count warnings after processing all chunks
- Removed unused `lambda_null` parameter from `calc_score_test`

## [2.8.1] - 2026-02-27

### Performance

- **NumPy grid REML/MLE vectorized**: replaced Python `for` loop over 50 grid lambdas
  with single `np.tensordot` call. Since all SNPs share the same lambda at each grid point,
  `Hi_eval` is `(n_grid, n_samples)` not `(n_snps, n_samples)`, eliminating the dominant
  memory allocation at scale. Benchmark (mouse_hs1940): Wald 18.3s → 6.4s (2.9x),
  All 34.4s → 11.6s (3.0x)
- `_fill_pab_recursion` uses `...` indexing to support both 3D and 4D Pab arrays,
  enabling the grid vectorization without duplicating the recursion logic

### Changed

- Extracted `_guard_P_yy` helper to deduplicate the P_yy clamping pattern (4 call sites)
- Extracted `_batch_grid_pab_numpy` to share tensordot + Pab computation between
  REML and MLE grid functions
- LOCO NumPy progress import hoisted to single location (was duplicated in pass 1/2)

### Fixed

- **LOCO NumPy runtime crash**: `progress_iterator` was imported from `jamma.utils`
  (which doesn't export it) instead of `jamma.core.progress` — caused `ImportError`
  when `show_progress=True` (the default)

## [2.8.0] - 2026-02-27

### Added

- **NumPy LOCO kinship streaming**: `_compute_loco_kinship_streaming_numpy()` — pure NumPy
  LOCO kinship computation (no JAX dependency), enabling `--loco --backend numpy` workflows
- **`LazySnpMeta` in `schema.py`**: Single canonical source for lazy PLINK metadata wrapper
  (was duplicated in `loco.py` and `runner_streaming.py`)
- **Shared LOCO helpers**: `_collect_chr_snp_stats()` and `_filter_chr_snps()` extract
  duplicated pass-1 SNP statistics and filtering logic from JAX/NumPy chromosome runners
- Backend validation in `run_lmm_loco`: raises `ValueError` for invalid backend values
- Write-offset validation in NumPy LOCO path: raises `RuntimeError` if pre-allocated
  result arrays are not fully written
- Diagnostic error handling around NumPy LOCO computation loop (logs chromosome, chunk
  offset, and SNP count on failure)
- **GEMMA covariate validation tests**: 4 mouse_hs1940 covariate tests for NumPy backend
  (Wald, LRT, Score, All modes) validating beta, SE, p-values against GEMMA reference
- **Synthetic no-covariate GEMMA validation tests**: LRT, Score, and All mode tests
  completing the NumPy backend validation matrix

### Changed

- `_P_YY_MIN` constant (1e-8) propagated from `likelihood.py` to `likelihood_numpy.py`
  and `stats.py` (was hardcoded in 7 locations)
- `runner_streaming.py` imports `LazySnpMeta` from `schema.py` instead of defining its own copy

### Fixed

- LOCO backend dispatch: `backend="numpy"` now uses NumPy kinship streaming instead of
  unconditionally importing JAX kinship module
- `pipeline.py` XLA profiling catch restored `AttributeError` (JAX can raise this on
  some platforms when profiling is unavailable)
- `backend.py` logger text: "to suppress this warning" → "to suppress this error" (matches
  actual log level)
- `generate_loco_fixtures.sh`: corrected GEMMA version reference (0.96 → 0.98.5)
- `test_loco.py`: fixed `ModuleNotFoundError` from invalid conftest import

## [2.7.1] - 2026-02-27

### Added

- **GEMMA LOCO integration test**: 3-chromosome validation (beta, SE, p_wald, l_remle,
  logl_H1, rank correlation, top hits) against GEMMA LMM with JAMMA-computed LOCO kinship
- Fixture generation scripts: `generate_loco_synthetic.py` (PLINK data),
  `generate_loco_fixtures.sh` (Docker-based GEMMA reference outputs)
- `logl_H1` per-chromosome comparison test (LOCO-04b)
- Merge completeness assertion in LOCO test fixture (detects inner-join data loss)

### Changed

- `load_phenotypes_from_fam` extracted to `conftest.py` for reuse; simplified to
  `np.loadtxt(usecols=5)`
- CI: dropped Intel Mac job (macos-13 deprecated) and Windows job (pytest-xdist
  deadlock); added `--cov-fail-under` per matrix entry
- Causal SNP check in `generate_loco_fixtures.sh` is now a hard failure (was warning)
- Tolerance rationale comments added to LOCO integration tests

### Fixed

- CI: added per-matrix `--cov-fail-under` thresholds (80% JAX, 50% NumPy-only)
- `pytest.importorskip('jax')` added before JAX-only imports in all
  JAX-dependent test files (fixes NumPy-only CI)

## [2.7.0] - 2026-02-26

### Added

- **Pure-NumPy backend**: Full LMM association (Wald, LRT, Score, All modes) without
  JAX dependency — `jamma` now works out-of-the-box on any platform with just numpy
- **`--backend` CLI flag**: Explicit backend selection (`auto`, `jax`, `numpy`); `auto`
  prefers JAX when available, falls back to NumPy
- **`backend` parameter on `gwas()` API and `PipelineConfig`**: Programmatic backend control
- **`special.py` module**: Pure-stdlib `betainc()` (Lentz continued-fraction) and `chi2_sf()`
  implementations — eliminates scipy dependency for p-value computation
- **`prepare_common.py`**: Shared null-model setup (eigendecomposition, rotation, REML)
  extracted from JAX-specific code for reuse by both backends
- **`likelihood_numpy.py`**: Batch Uab/Pab/REML/MLE computation and Wald/LRT/Score
  statistics using pure NumPy — vectorized across grid/refinement steps
- **`compute_numpy.py`**: Mode-dispatch layer routing to NumPy likelihood functions
- **`runner_numpy.py`**: Streaming chunk-loop LMM runner using NumPy backend with
  identical output format to JAX runner
- **`detect_backend()` and `log_backend_selection()`**: Backend probing and diagnostic logging
- **Platform-smart JAX defaults**: `pip install jamma[jax]` auto-includes JAX on Linux
  and ARM Mac via PEP 508 markers; Windows/Intel Mac get NumPy-only by default
- **`requires_jax` pytest marker**: JAX-dependent tests auto-skip when JAX unavailable
- **Cross-backend CI matrix**: Tests run on Linux+JAX, Linux+NumPy, macOS+JAX,
  Windows+NumPy, and Linux+JAX(3.11) configurations
- **406 new tests** in `test_special.py` for `betainc`/`chi2_sf` edge cases
- **Typed backend literals**: `BackendRequest` and `BackendResolved` types for pipeline config

### Changed

- JAX moved from required to optional dependency (`jamma[jax]` extra)
- All `__init__.py` modules guard JAX imports behind `has_jax()` — `import jamma`
  succeeds without JAX installed
- `PipelineConfig.backend` uses `BackendRequest` literal type; `PipelineResult.backend`
  uses `BackendResolved` literal type
- `conftest.py` registers `requires_jax` marker and auto-applies to JAX-importing tests
- Dockerfile updated for layered `jamma[jax]` install
- `_compute_lmm_chunk` defaults aligned: `n_grid=50`, `n_refine=10` (was inconsistent
  between JAX and NumPy compute modules)
- `snp_filter.py` `np.errstate` scope narrowed to `invalid`/`divide` only (was `all`)

### Fixed

- **`has_jax()` swallowed `RuntimeError`/`OSError`**: JAX installation failures (broken
  CUDA, missing libraries) now log a warning instead of silently returning `False`
- **`runner_jax.py` crashed on `kinship=None`**: Type signature and guard updated to
  accept `None` when pre-computed eigendecomposition is provided
- **Missing eigenpair validation in `runner_jax.py`**: Added dimension checks matching
  `runner_numpy.py` — catches shape mismatches before LAPACK calls
- **`prepare.py` dropped `TypeError`**: `_setup_cpu_sharding` exception tuple restored
  to include `TypeError` alongside `RuntimeError`/`ValueError`
- **Silent invalid `lmm_mode` in `_compute_null_model_common`**: Mode 1 now returns
  `None` explicitly; invalid modes raise `ValueError` (was silently returning `None`)
- **`betainc` ArithmeticError catch unlogged**: CF non-convergence now logged at debug level
- **All-SNPs-filtered produced silent empty return in `runner_jax.py`**: Now logs warning
- **Memory estimate ran unconditionally in `runner_jax.py`**: Now gated behind `check_memory` flag
- **P_yy zero in Score test denominator**: Clamped to 1e-8 floor to prevent Inf F-statistic
- **`runner_numpy.py` missing early validation**: Raises `ValueError` when neither kinship
  nor eigendecomposition is provided
- **`_RESULT_FIELDS` import path**: `runner_jax.py` now imports from `schema.py` (was
  importing from deleted `results.py` path)

## [2.6.1] - 2026-02-26

### Fixed

- `test_lmm_jax_chunk_invariance` passed consumed kinship to second
  `run_lmm_association_jax` call (in-place eigendecomp overwrites K with
  eigenvectors; added `.copy()`)

## [2.6.0] - 2026-02-26

### Added

- **Runtime buffer mismatch detection**: If `eigh_lo` ignores the `out=` parameter
  (future numpy change), `INPLACE_EIGEN_AVAILABLE` flag is set False at runtime
  and memory estimates automatically correct to include separate eigenvector allocation
- Tests for buffer mismatch flag update, fallback memory estimates, guard clauses,
  safety margin cap, ImportError logging in `_inplace_eigen_available()`
- Tests for LOCO kinship bugs: aliasing, chromosome ordering, fallback normalization,
  n_filtered=0 guard, GeneratorExit partial retention, flush failure propagation,
  `_dsyevd_workspace_gb` formula (LIWORK uses 8-byte integers)

### Changed

- `chr_sort_key` extracted from `loco.py` to `utils/__init__.py` (DRY — used by
  both loco.py and kinship/compute.py); unknown chromosome sentinel raised from
  100 to 1000 (supports species with >99 numeric chromosomes)
- Memory safety margin capped at 10GB absolute (was unbounded 10%, which
  demanded 50GB+ headroom at scale)
- Memory estimates adapt to in-place vs fallback eigendecomp path at runtime
- `IncrementalAssocWriter.__exit__` cleans up partial output on any `Exception`
  subclass (was OSError-only); retains partial output on `KeyboardInterrupt`,
  `SystemExit`, and `MemoryError` (partial results are valid up to point of failure)
- Docstrings clarified: K "may be overwritten" / "treat as consumed" (was
  unconditional "OVERWRITTEN" which was inaccurate for fallback path)
- `_inplace_eigen_available()` ImportError logged at warning (was info) — indicates
  broken installation
- In-place eigendecomp fallback logged at warning (was info) — 320GB impact at scale
- Unknown chromosome names logged at info (was debug) — aids debugging LOCO issues
- `pipeline.py` XLA profiling catch narrowed to `(OSError, ImportError, AttributeError)`
  (was bare `except Exception`)
- `S_full_np` marked read-only after in-place division in `_yield_loco_matrices`
  to guard against accidental re-mutation
- LOCO `write_kinship_matrix` error includes chromosome name and path for diagnostics

### Fixed

- **`IncrementalAssocWriter.__exit__` flush failure silently deleted output** — now
  raises after cleanup so callers know the write failed (was `logger.warning` + return)
- `_format_duration` produced "2h 60m" for durations near hour boundaries and
  "60 min" at exactly 3599s due to `:.0f` rounding (now uses integer truncation
  throughout)
- README Low-level API example passed consumed kinship matrix to streaming runner
  (now correctly passes eigenvalues/eigenvectors); added missing `import numpy as np`
- `test_runner_jax.py` passed mutated K as kinship to runner (added `.copy()`)
- `_yield_full_kinship_fallback` held persistent `K_full` alongside `S_full_np`
  while consumer processed yielded matrix (3 n×n matrices live). Now divides
  `S_full_np` in-place once and yields `.copy()` per chromosome (2 matrices
  live at yield: modified `S_full_np` + the copy), matching the LOCO memory
  gate budget
- Stale field comments on `MemoryBreakdown.sufficient` and
  `StreamingMemoryBreakdown.sufficient` (referenced old `total * 1.1` formula)
- Inaccurate comments: plink.py "two boolean ops" (actually 3), eigen.py
  "re-imports each call" (reads module attribute), CHANGELOG fallback description

## [2.5.8] - 2026-02-25

### Changed

- In-place kinship accumulation (`K += np.matmul(...)`) eliminates one n×n temporary per batch
- Size-gated eigendecomp symmetry check: full `np.allclose` for n<10k, vectorized sampled check for n≥10k (avoids 80GB temporary at 100k samples)
- Single-pass eigenvalue post-processing with in-place thresholding (no `np.where` allocation)
- LOCO valid_mask guard: skips n×n subsetting copy when all samples are valid
- LOCO SNP-list restriction uses precomputed boolean mask instead of per-chromosome `np.isin`

## [2.5.7] - 2026-02-23

### Added

- Unit tests for likelihood_jax.py edge cases: negative P_yy, degenerate SNPs, near-zero eigenvalues, lambda bounds, JAX/NumPy consistency, covariate rank validation, kinship symmetry checks
- CI coverage threshold (`--cov-fail-under=80`) enforced on the full test suite

### Fixed

- JAX REML and MLE paths now guard negative P_yy → NaN (previously only the NumPy path had this guard)
- CLI rejects `--mem-budget <= 0` with a clear error instead of silently proceeding
- Covariate rank validation: rank-deficient covariate matrices now raise `ValueError` before LMM runs
- Kinship eigendecomposition warns when input matrix is asymmetric
- LOCO warns and uses full kinship for chromosomes with 0 ksnps (was silently skipping them)
- Out-of-place kinship accumulation (`K = K + matmul(...)`) for deterministic FP rounding
- Explicit `del chunk` in streaming stats loops to free memory between iterations
- Test tolerances aligned with GEMMA_EQUIVALENCE.md (kinship 1e-10 → 1e-8)
- CI: `test-slow` job skips coverage threshold (partial test runs can't meet 80%)

### Changed

- Dockerfile: consolidated RUN layers, added non-root user (`jamma`, uid 1000)
- CI: upgraded `astral-sh/setup-uv` v4 → v5
- Pinned `ruff>=0.15.0` in dev dependencies to match CI
- Kinship non-LOCO path converted from JAX to numpy (JAX not initialized during kinship phase)
- Extracted `DevicePlacement` and shared chunk preparation into `prepare.py`
- Deferred JAX backend initialization until LMM phase
- Wall clock time estimates for kinship, eigendecomp, and LMM phases

## [2.5.6] - 2026-02-22

### Fixed

- LMM rotation (`U.T @ G`) now uses all physical cores instead of `physical_cores // n_jax_devices` — same bug class as eigendecomp (v2.5.4), but in the per-chunk dgemm. On a 48-core machine with 24 JAX devices, rotation ran with 2 threads instead of 48 (~16x slowdown per chunk, ~4 hours instead of ~30 minutes for 125k samples)
- Applied fix to all three runners: `runner_jax.py`, `runner_streaming.py`, `loco.py`
- Extracted `get_physical_core_count()` helper in `core/threading.py` to consolidate physical core detection (replaces inline `psutil.cpu_count(logical=False)` in eigen.py)

## [2.5.5] - 2026-02-22

### Added

- Regression tests for worker cap (verifies 32-worker limit on high-core machines)
- Regression tests for eigendecomp threading (verifies all physical cores used, not divided by JAX devices)
- Chunk sizing tests at Databricks scale (125k samples, parametrized across 1-48 devices)
- Fast synthetic LOCO partition tests (no fixture dependency)
- Eigendecomp memory gate integration tests (verifies MemoryError before LAPACK runs)

## [2.5.4] - 2026-02-22

### Fixed

- Eigendecomposition now uses all physical cores instead of `physical_cores // n_jax_devices` — JAX isn't running during `eigh`, so the thread reduction was a ~16x slowdown on multi-device configs

## [2.5.3] - 2026-02-22

### Fixed

- Cap matrix writer workers at 32 (was unbounded cpu_count) — 96 workers on Databricks added process overhead with no I/O benefit
- Eliminate per-row `tuple()` allocation in worker formatting — was creating 125k Python float objects (~3 MB) per row per worker, causing GC thrashing
- Correct peak disk estimate to account for all chunks existing simultaneously during worker phase

## [2.5.2] - 2026-02-22

### Fixed

- Matrix writer no longer fills `/tmp` when writing large kinship matrices — temp files (memmap + chunks) are now created on the same filesystem as the output file
- Chunks are deleted eagerly during concatenation, reducing peak disk from 2x output size to ~1x
- Memmap is freed before concatenation starts, reclaiming matrix-sized temp space earlier
- Pre-flight disk space warning when free space looks insufficient for the write

## [2.5.1] - 2026-02-22

### Added

- PyPI keywords and classifiers for search discoverability
- Project URLs (Homepage, Repository, Documentation, Changelog, Issues) for PyPI verified details

## [2.5.0] - 2026-02-21

### Added

- **CPU device sharding**: JAX automatically partitions SNP batches across
  virtual CPU devices using `NamedSharding`. Auto-configures as
  `max(1, physical_cores // 2)` — no user action required. Override with
  `JAMMA_JAX_DEVICES` environment variable for custom tuning.
- **BLAS thread coordination**: BLAS thread count auto-reduces when multiple
  JAX devices are active to avoid oversubscription. Override with
  `JAMMA_BLAS_THREADS` environment variable.
- **`--profile-dir` CLI flag**: Capture XLA profiling traces for
  TensorBoard/Perfetto analysis. Degrades gracefully — profiling failures
  never prevent GWAS results.
- **Per-stage timing**: LMM runners now log timing breakdowns for
  eigendecomposition, DGEMM rotation, JAX compute, and result writing.
- **JAX profiler annotations**: `TraceAnnotation` labels on all pipeline
  stages for use with `--profile-dir`.
- **Benchmark harness**: `pytest-benchmark` pedantic-mode benchmarks for
  eigendecomp, DGEMM rotation, JAX optimization, and full pipeline on
  mouse_hs1940. Includes hardware context (CPU model, BLAS backend, device
  count) for cross-machine comparison.
- **Hardware context module**: `jamma.core.hardware.get_hardware_context()`
  collects CPU, BLAS, JAX, and platform info for benchmark reproducibility.
  `assert_x64_precision()` guard prevents silent float32 fallback in
  benchmark entry points.

### Fixed

- **Sharding divisibility fallback**: SNP counts not evenly divisible by the
  device count (e.g. 50,000 SNPs with 32 devices) no longer silently disable
  sharding. UtG arrays are zero-padded to the next device-count multiple and
  padded results are discarded.
- **Chunk device alignment**: `_compute_chunk_size` and `auto_tune_chunk_size`
  round chunk sizes to device-count multiples, preventing XLA from padding
  partial shards internally.

## [2.4.5] - 2026-02-20

### Fixed

- **Assert→RuntimeError**: Replaced `assert` with `if`/`raise` for write_offset
  check in JAX runner — `assert` is stripped under `python -O`, risking silent
  data truncation.
- **File handle leak**: Fixed `IncrementalAssocWriter.__exit__` skipping file
  close on non-OSError exceptions.
- **Off-by-one in retry count**: Error message now reports correct attempt number.
- **Exception propagation**: `verify_jax_installation()` re-raises original
  exceptions instead of wrapping in `RuntimeError`.

### Added

- **Eigenvector shape validation**: Raises `ValueError` when pre-computed
  eigenvectors don't match sample count after filtering.
- **JAX int32 overflow detection**: Streaming and LOCO runners catch and log
  diagnostic context for JAX buffer overflow errors.
- **Parameter validation**: `_compute_lmm_chunk()` validates `logl_H0` and
  `Hi_eval_null` are provided for modes that require them.
- **23 new tests**: Covers `_safe_sqrt` boundary behavior, `_clamp_p_yy`
  clamping, P_yy in log-likelihood, over-parameterization guard, and golden
  section optimizer.

### Changed

- **DRY P_yy clamping**: Extracted `_clamp_p_yy()` helper replacing 4 duplicate
  clamping blocks in likelihood.py.
- **Precomputed sample filter**: `needs_sample_filter` flag computed once before
  hot loops instead of `np.all(valid_mask)` per iteration.
- **Narrowed eigendecomp exception**: Catches `np.linalg.LinAlgError` before
  generic `Exception` with PSD-specific guidance.
- **Debug tracebacks**: CLI exception handlers log `exc_info=True` for verbose
  diagnostics.

## [2.4.4] - 2026-02-19

### Added

- **GEMMA-style startup banner**: Consolidated dataset summary logged at startup for
  both LMM and kinship modes — version, release date, total/analyzed individuals,
  covariates, phenotypes, and total SNPs.
- **Auto-derived release date**: Hatchling build hook (`hatch_build.py`) embeds the git
  commit date into the package at build time. No manual maintenance required — the date
  appears in the banner and `--version` output automatically.

## [2.4.3] - 2026-02-19

### Changed

- **File-to-file parallel writer**: Workers write formatted text to per-chunk temp
  files instead of returning ~1.2 GB bytes objects through the multiprocessing IPC
  pipe. Eliminates memory spike at end of write phase — at 100k×100k with 16 workers,
  old code buffered ~19 GB in the IPC queue; new code: ~0 bytes in IPC.
- **Removed 16-worker cap**: `write_matrix_parallel()` defaults to `cpu_count`
  instead of `min(cpu_count, 16)`. Per-worker memory is now ~150 MB process overhead
  (not 1.2 GB buffered bytes), so higher worker counts are safe.

## [2.4.2] - 2026-02-18

### Fixed

- **Loguru traceback in pool errors**: Use `logger.opt(exception=e)` instead of
  `exc_info=True` (stdlib pattern ignored by Loguru) for full traceback on worker failures.
- **Improved error handling**: Use `RuntimeError` for worker exception wrapping
  (fixes fragile `type(e)(...)` pattern), log warning on temp file cleanup failure,
  add `TMPDIR` hint for disk-full errors.
- **Pre-commit hooks**: Fix hook chain so ruff lint/format runs in CI and locally
  (was bypassed by beads `core.hooksPath` override).

### Changed

- **Code simplification**: List comprehensions in parallel matrix writer, extracted
  shared test fixture for temp dir isolation.

## [2.4.1] - 2026-02-18

### Fixed

- **Docker SIGBUS on large matrices**: Replaced `SharedMemory` (POSIX `shm_open()`)
  with file-backed `numpy.memmap` in `write_matrix_parallel()`. Docker defaults
  `/dev/shm` to 64 MB — a 100k×100k float64 matrix is ~75 GB, causing SIGBUS on
  access. The memmap approach uses filesystem-backed temp files instead, bypassing
  `/dev/shm` entirely. ([cpython#114390](https://github.com/python/cpython/issues/114390))

## [2.4.0] - 2026-02-18

### Added

- **Parallel matrix writer**: `write_matrix_parallel()` using `multiprocessing.Pool.imap`
  with SharedMemory to format matrix rows across CPU cores. Falls back to `np.savetxt`
  for small matrices (<500 rows). Byte-identical output to `np.savetxt` for all sizes.
  Reduces 100k×100k matrix write from ~30min to ~2-4min.
- **Unified output schema**: `schema.py` with `StatColumn`/`ModeSpec` frozen dataclasses
  as single source of truth for LMM output column definitions. `MODE_SPECS` frozen via
  `MappingProxyType`. Replaces 4 separate dispatch tables across 3 modules.
- **Fast PLINK line counting**: `_count_lines_fast()` uses binary `bytes.count(b'\n')`
  in 1MB chunks instead of text-mode `sum(1 for _ in f)`. Correctly handles files
  without trailing newline.
- **`write_arrays_batch` hot path**: Formats and writes results directly from numpy
  arrays, bypassing `AssocResult` construction. Validates stat array lengths and
  snp_info keys upfront.

### Changed

- Kinship and eigenvector writers now delegate to `write_matrix_parallel()`
- `IncrementalAssocWriter` retry logic consolidated into shared `_write_buf()` method
- `ACCUM_KEYS`, `RESULT_FIELDS`, `FORMAT_COLUMNS`, `HEADERS`, `TEST_TYPE_MAP` derived
  mechanically from `MODE_SPECS` (eliminates manual sync)
- File I/O functions in `kinship/io.py` and `lmm/eigen_io.py` log resolved paths

### Fixed

- Partial file cleanup on worker failure in parallel matrix writer
- `n_workers` validation (reject < 1) in `write_matrix_parallel`
- `StatColumn.fmt` included in string type validation
- `ModeSpec` validates duplicate header names across columns
- Worker error context includes chunk row range for debugging
- Pool errors logged before `pool.terminate()` for diagnostics

## [2.3.0] - 2026-02-18

### Fixed

- **Lambda bounds not plumbed to null MLE**: `l_min`/`l_max` now passed through
  `_compute_null_model` to `compute_null_model_mle` so null-model optimization
  respects user-configured lambda bounds
- **Memory check used raw PLINK dimensions**: Pipeline memory estimation now uses
  post-filter sample count (`n_valid`) and actual covariate count instead of raw
  `.fam`/`.bim` metadata dimensions
- **LOCO accumulated full-chromosome results**: Results now flushed per disk chunk
  instead of accumulating all JAX arrays for the entire chromosome before conversion
- **`chunk_size <= 0` in `stream_genotype_chunks`**: Guard against `chunk_size=0`
  (ZeroDivisionError) and negative values (infinite range)
- **Batch runner missing `jax.clear_caches()`**: Added after chunk loop for parity
  with streaming runner — prevents JIT trace accumulation across LOCO runs
- **Batch runner missing lambda boundary tracking**: Diagnostic warning for SNPs
  converging at lambda bounds (was only in streaming runner)
- **Output prefix path traversal**: `OutputConfig` and `PipelineConfig` now reject
  `output_prefix` containing path separators
- **Biological chromosome ordering in LOCO**: Chromosomes now sort 1..22, X, Y,
  XY, MT instead of lexicographic order (1, 10, 11, ..., 2, 20, ...)
- **CLI timing key access**: Bare `result.timing["key"]` replaced with `.get()`
  to prevent KeyError when timing keys are missing
- **CLI `n_covariates` display**: Pipeline now populates `n_covariates` in timing
  dict (was always showing default value)
- **LRT validation `all_passed`**: Beta/SE NaN validation now included in LRT
  comparison (was silently skipped)
- **Duplicate BIM SNP IDs**: `resolve_snp_list_to_indices()` now warns about
  duplicate SNP IDs and keeps first occurrence (was silently using last)
- **`donate_argnums` deprecation**: Removed deprecated JAX `donate_argnums` from
  golden section optimizers
- **Empty samples guard**: `run_lmm_association_jax()` now raises `ValueError`
  when all phenotypes are NaN/-9 (no valid samples remain)
- **Streaming runner empty samples guard**: Streaming runner raises `ValueError`
  on zero valid samples after filtering
- **`__main__.py` missing `__name__` guard**: Prevented double execution on import
- **`ensure_jax_configured` silent on conflicts**: Now raises `RuntimeError` on
  conflicting non-default args after JAX is locked (was silent warning)
- **Negative P_yy only logged at debug**: Elevated to `warning` with lambda context
  in 4 locations in `likelihood.py`
- **GPU fallback only logged at debug**: Elevated to `warning` in `prepare.py`
- **Empty results misclassification**: `compare.py` guarded `all()` on empty lists
- **Double eigendecomp in `test_hypothesis.py`**: Reduced to single `eigh` call

### Changed

- `_MAX_BUFFER_ELEMENTS` derived from `INT32_MAX` constant instead of magic number
- `_LazySnpMeta.__getitem__` supports slice indexing for list-like behavior
- Removed unused backward-compat re-exports from `runner_jax.py`
- Migrated `snp_filter.py` logging from `print()` to `loguru`
- SNP statistics in streaming runners use numpy arrays instead of `locals()` dict
- Replaced `np.random.seed()` with `np.random.default_rng()` in tests
- Dead code removed: unused `n_snps` param, unreachable shape check, unused
  `lambda_val` param, 8 duplicated `setup_jax` test fixtures
- 4 `format_assoc_line_*` → 1 table-driven function (`io.py`, -161 lines)
- 4 `_build_results_*` → 1 with `_RESULT_FIELDS` dispatch (`results.py`)
- `runner_jax.py` mode-to-arrays refactored to use `_RESULT_FIELDS` (DRY)
- Header selection unified to table-driven `_HEADERS` dict in `io.py`
- Input validation on dispatch keys in `io.py` and `results.py`
- Fixture paths use `Path(__file__).parent` instead of cwd-relative (6 files)
- CLI subprocess tests decoupled from `uv` runtime
- `ToleranceConfig` gains `p_lrt_rtol` field
- Pinned ruff to 0.15.x across local dev deps and CI pre-commit

### Added

- Streaming-vs-batch parity tests for degenerate SNP and empty-samples edge cases
- Hypothesis property tests for variance computation and SNP filtering
- `Raises` docstring for `run_lmm_association_jax()` ValueError
- `__main__.py` for `python -m jamma` execution
- LOCO integration tests: multi-pass batching, NaN covariates, MAF filtering
- Streaming LRT/Score mode tests, writer retry/rollback tests
- 7 new Hypothesis property tests for Score test and LRT invariants
- Tier markers on all 723 tests (392 tier0, 309 tier1, 22 tier2)
- 22 unit tests in `test_review_fixes.py` for dispatch validation and erfc

## [2.2.0] - 2026-02-17

### Added

- **Lambda bounds** (`-lmin`/`-lmax`): Configurable optimization bounds for lambda
  with boundary convergence warnings when SNPs cluster at bounds
- **Individual weights** (`-widv`): GEMMA-exact kinship pre-transformation
  K[i,j] /= sqrt(w_i * w_j) via memory-efficient two-pass scaling (O(n) memory)
- **Categorical covariates** (`-cat`): One-hot encode specified covariate columns
  with reference level dropped. JAMMA-specific feature (not GEMMA's -cat)
- `-wsnp` flag accepted (hidden, not yet implemented — clear error message)
- Eigen I/O validation: empty file checks, parse error wrapping with file paths,
  `atleast_1d`/`atleast_2d` for single-line files, square matrix validation

### Changed

- `IncrementalAssocWriter`: retries transient write failures with backoff,
  truncates partial writes before retry, cleans up partial files on final failure
- Replaced `click.echo` with `loguru` in I/O module (removes click dependency from io)
- Eigen file writers use `np.savetxt` instead of Python f-string loops
- Slow gwas_api integration tests marked `@pytest.mark.slow`, skipped by default

### Fixed

- Categorical single-level columns with NaN now keep a NaN marker column
  (previously deleted entirely, losing missingness signal for pipeline filtering)
- Weight file reader rejects multi-column files instead of silently flattening
  via `.ravel()` (prevented weight misalignment)
- Weight file reader rejects NaN values (bypassed all scaling logic due to
  NaN comparison semantics)
- `__exit__` cleanup now properly nulls `_file` on successful close
- Writer retry truncates partial writes to prevent duplicate lines on retry

## [2.1.0] - 2026-02-16

### Added

- **Multi-pass LOCO S_chr batching**: When all per-chromosome S_chr matrices don't
  fit in memory (e.g. 100k samples x 22 chromosomes), chromosomes are automatically
  batched across multiple disk passes — S_full computed once and reused
- **LOCO writer passthrough**: `_run_lmm_for_chromosome` streams results directly
  to disk via optional `writer` parameter, eliminating per-chromosome result
  accumulation in memory
- **In-memory mode warnings**: Log warning when running without `output_path` with
  >100k SNPs, recommending disk streaming
- Memory estimates now logged even when `check_memory=False`
- **CONTRIBUTING.md**: Development setup, testing, code style, and PR guidelines

### Changed

- **CLI: Typer → Click**: Flat GEMMA-compatible CLI — `jamma -gk 1 -bfile data` instead
  of `jamma gk -bfile data`. True drop-in replacement for GEMMA command lines
- **Dockerfile**: Uses uv for package management; documents `--platform linux/amd64`
  requirement for MKL (x86_64-only)
- All documentation updated to flat CLI syntax
- LOCO kinship: extracted `_stream_s_full_and_chr` and `_yield_loco_matrices`
  helpers to eliminate code duplication across single-pass and multi-pass paths
- Deduplicated `_yield_chunk_results` call in `_run_lmm_for_chromosome` — iterator
  created once, only consumption differs (writer vs list)
- Memory safety margin reduced from 50% to 10% for streaming kinship
- **Removed Databricks notebooks and Dockerfile**: Moved to separate `jamma-databricks`
  project — JAMMA repo now contains only the library and a general-purpose Dockerfile
- Minimum numpy bumped to 2.0+ (1.26 is EOL)

### Fixed

- JAX device array leak on write exception — `eigenvalues_jax`, `UtW_jax`,
  `Uty_jax` now freed via `try/finally` in `_run_lmm_for_chromosome`
- Multi-pass memory accounting underestimated first-pass peak by one `matrix_gb`
  (JAX and numpy S_full coexist briefly during conversion)
- Exception-safe writer lifecycle using `ExitStack` for LOCO writer
- Eigen file validation against covariate-filtered sample count
- Empty output and dead `logls_mle` accumulation removed from LMM runner
- `check_memory` flag now respected in `eigendecompose_kinship`

### Performance

- Two-pass chunked column iteration in LOCO replaces single full-matrix read
- Lazy SNP metadata loading and early cleanup of pass-1 statistics arrays
  (`all_vars`, `all_means`, `all_miss_counts`) immediately after deriving filters
- Free kinship matrix after `write_eigen` instead of holding until end
- Remove unnecessary `U.T` contiguous transpose copy
- Hoist `snps_indices` set conversion out of LOCO chromosome loop
- Skip `impute_and_center` in multi-pass when no target chromosomes in chunk

## [2.0.0] - 2026-02-12

### Added

- **LOCO kinship** (`-loco` flag): Leave-one-chromosome-out kinship via streaming
  subtraction approach — computes per-chromosome K_loco one at a time for memory
  efficiency. Eliminates proximal contamination in LMM association
- **Eigendecomposition reuse** (`-d`/`-u`/`-eigen` flags): Save and load pre-computed
  eigendecomposition for multi-phenotype workflows — skip O(n³) eigendecomp after first run
- **Phenotype selection** (`-n` flag): Select phenotype column from multi-phenotype
  .fam files (1-based indexing, matching GEMMA)
- **Standardized kinship** (`-gk 2`): GEMMA-compatible standardized relatedness matrix
  using (X - mean) / sqrt(p*(1-p)) normalization
- **SNP subset selection** (`-snps`/`-ksnps` flags): Restrict association testing and/or
  kinship computation to SNP lists (one RS ID per line)
- **HWE QC filtering** (`-hwe` flag): Hardy-Weinberg equilibrium chi-squared
  goodness-of-fit test — exclude SNPs below p-value threshold. Genotype counts
  piggyback on pass-1 streaming (no extra disk pass)
- **PLINK dimension validation**: Cross-validate .bed file size against .fam/.bim
  line counts before processing
- **Genotype value validation**: Warn on values outside expected range {0, 1, 2, NaN}
- **`apply_snp_list_mask()` helper**: DRY bounds-validated SNP mask application
  (replaces 3 duplicate code blocks in kinship and LMM runners)
- **SNP filter regression tests**: Verify searchsorted-based chunk filtering matches
  naive linear scan across edge cases (boundary SNPs, full/empty chunks, single-element)
- **Missingness test suite**: Heterogeneous missingness patterns, column-specific
  imputation accuracy, edge cases (all-missing, no-missing, single-sample)
- **Hypothesis property tests for v2.0 features**: 14 new tests covering HWE chi-squared
  (p-value bounds, allele swap symmetry, perfect equilibrium, degenerate inputs,
  vectorized/scalar equivalence), standardized kinship (symmetry, PSD, trace approximation,
  shape consistency), and eigen I/O round-trip (.10g format reconstruction, orthonormality,
  eigenvalue precision). Total: 42 hypothesis tests (up from 29)

### Changed

- **Streaming SNP filtering**: Replaced O(n) linear scan with `np.searchsorted` for
  chunk-level SNP range filtering — eliminates per-SNP Python overhead in streaming runners
- **Memory module comments**: Updated docstrings to reflect streaming architecture
  and actual component breakdown
- **HWE accumulators**: Upgraded int32 → int64 for overflow safety on large cohorts
- **HWE NaN handling**: Replaced `np.nan_to_num` with explicit `np.where` to avoid
  silent inf/neginf clobbering

### Fixed

- **HWE silently ignored in LOCO mode**: `-hwe` parameter was accepted but had no
  effect when `-loco` was active — now rejected with clear error message
- **CLI gk ksnps errors uncaught**: Missing/invalid ksnps file produced a traceback
  instead of user-friendly error — now wrapped in try/except
- **HWE threshold >1.0 accepted**: Out-of-range p-value threshold now validated

### Removed

- **Bioconda recipe**: Removed `bioconda/meta.yaml` and automated bioconda PR submission —
  bioconda's conda-forge numpy is LP64 only, which silently breaks for JAMMA's target
  users (>46k samples require ILP64 MKL). pip is the canonical install path.

## [1.5.1] - 2026-02-10

### Changed

- README logo and badge layout refinements

## [1.5.0] - 2026-02-10

### Added

- **PipelineRunner service**: Shared orchestration class eliminates duplicated pipeline
  logic between CLI and Python API — single source of truth for validate, parse, check
  memory, load kinship, load covariates, run LMM
- **Bioconda recipe**: `bioconda/meta.yaml` and automated PR submission to
  bioconda-recipes on each release
- **Memory/chunk coupling**: Memory estimation now uses computed chunk size from
  `_compute_chunk_size()` instead of hardcoded 10,000 — estimates match actual runtime
- **README badges**: Bioconda, JAX, NumPy, Hypothesis
- **Project logo** in README hero section

### Changed

- CLI `lmm` command delegates to `PipelineRunner` (256 → 78 lines)
- `gwas()` API delegates to `PipelineRunner` (164 → 28 lines)
- Removed import-time side effects — `configure_jax()` is now lazy via
  `ensure_jax_configured()` sentinel pattern
- CI restructured into 3 jobs: `lint`, `test-fast` (unmarked tests), `test-slow`
  (tier2/slow, master-only)
- Ruff pre-commit hook updated v0.8.6 → v0.15.0
- Publish workflow updated for live PyPI with automated bioconda PR submission

### Fixed

- Memory estimates used hardcoded chunk size (10,000) instead of the actual computed
  chunk size — could over/underestimate by 2-5x at different scales

## [1.4.3] - 2026-02-10

### Added

- **Production-scale GEMMA validation**: 85,000 real samples × 91,613 SNPs — 100%
  significance agreement, 100% effect direction agreement, Spearman rho 1.000000
- **Compare-only mode** for GEMMA comparison notebook — load pre-computed results
  from configurable source paths, skip all compute
- **OOM-safe kinship comparison**: Sampled Spearman (10M elements) + chunked row-by-row
  statistics for 85k+ matrices without materializing `np.triu_indices` (~58GB) or
  full rank arrays (~60GB)
- **Performance documentation** (`docs/PERFORMANCE.md`): Bottleneck breakdown,
  theoretical floor analysis, configuration guide, validation results
- **Top-level `gwas()` API**: Single-call entry point for full GWAS pipeline
  - `from jamma import gwas` — load data, compute kinship, run LMM, write results
  - Returns `GWASResult` dataclass with associations, timing, and summary stats
  - Supports pre-computed kinship, covariates, save-kinship mode
- **Phase-specific memory estimation**: `estimate_lmm_memory()` and
  `estimate_lmm_streaming_memory()` check only LMM-phase memory (not full pipeline peak)
- **Progress bar** for in-memory kinship computation
- **Method logging** for kinship computation (in-memory vs streaming)

### Changed

- LMM runners use phase-specific memory checks instead of total pipeline peak —
  fixes false `MemoryError` when eigendecomp is already complete (e.g., 100k sample
  benchmark: 300GB available, LMM needs ~96GB, was incorrectly demanding 320GB)
- `__version__` now reads from package metadata (`importlib.metadata`) instead of
  hardcoded string — stays in sync with `pyproject.toml` automatically
- JAX cache directory creation wrapped in `try/except OSError` — no longer crashes
  in restricted environments (read-only filesystems, containers)
- Memory safety margin reduced from 50% to 10% based on empirical benchmarks
- Extracted shared helpers in memory estimation (`_check_available`,
  `_streaming_component_sizes`) to reduce duplication
- Vectorized phenotype parsing in `gwas.py` (numpy ops instead of list comprehension)
- Vectorized per-SNP imputation in streaming runner (~2x faster)
- GEMMA comparison notebook writes output to local `/tmp/` instead of DBFS FUSE
- GEMMA comparison notebook accepts pre-existing GEMMA output files

### Fixed

- **LMM MemoryError at 100k samples**: LMM phase demanded 320GB (eigendecomp peak)
  against 300GB available, but only needed ~96GB. Now uses `estimate_lmm_memory()`
- **JAX async dispatch**: `block_until_ready()` in kinship compute loop — progress
  bars and timing now reflect actual compute, not async dispatch time
- **Progress bar lifecycle**: Bars complete cleanly (no hanging on final iteration)
- **Double `.bed` extension**: Fixed `.bed.bed` path construction in GEMMA comparison notebook
- Flaky `test_gwas_with_precomputed_kinship` timing assertion under pytest-xdist

## [1.3.0] - 2026-02-07

### Added

- **Golden section optimizer**: Replaced Brent's method (via scipy) with grid search +
  golden section refinement for lambda optimization — removes scipy runtime dependency
- Auto-select streaming kinship for large datasets (>10k samples)

### Changed

- **Removed scipy runtime dependency**: scipy is now dev-only (tests use `scipy.stats`).
  JAMMA uses `numpy.linalg.eigh` for eigendecomposition, which correctly uses ILP64
  when numpy is built with ILP64 MKL
- Deleted `optimize.py` — lambda optimization now lives in `likelihood_jax.py`
- Stripped numba from `likelihood.py`
- Split `runner_streaming.py` from `runner_jax.py` (separate module)
- Extracted shared utilities: `prepare.py`, `chunk.py`, `results.py`, `progress.py`,
  `snp_filter.py`
- Cached contiguous `U.T` in both LMM runners (perf)
- Replaced list accumulators with pre-allocated numpy arrays (perf)

### Removed

- `optimize.py` (Brent's method via scipy)
- Numba dependency in likelihood computation
- scipy as a runtime dependency

### Fixed

- `NotImplementedError` for kinship mode 2 (standardized) — now raises explicitly
  instead of producing wrong results

## [1.2.0] - 2026-02-05

### Added

- **Databricks benchmark notebook** (`notebooks/databricks_jamma_vs_gemma.py`):
  Widget-parameterized notebook comparing JAMMA vs GEMMA runtime and accuracy
- **Kinship matrix comparison**: Spearman rho, Frobenius norm, max/mean absolute/relative diff
- **CPU pinning for GEMMA**: `taskset --cpu-list 0-23` for eigendecomp in benchmark notebook

### Changed

- Skip JIT warmup for large datasets (>10k samples) to avoid double eigendecomp
- Auto-select streaming kinship for large datasets (>10k samples) with progress bar
- Expanded WHY_JAMMA.md with detailed GEMMA vs JAMMA speed comparison

### Fixed

- Double eigendecomposition in benchmark notebook (warmup was running full pipeline)

## [1.1.0] - 2026-02-05

### Added

- **Score test** (`-lmm 3`): Efficient screening test using null model lambda
- **Likelihood ratio test** (`-lmm 2`): MLE-based chi-square test
- **All tests mode** (`-lmm 4`): Combined Wald, LRT, and Score output
- **Covariate support**: `-c <file>` flag for covariate file input (GEMMA format)
- **Memory pre-flight checks**: Fail fast before OOM instead of silent crash
  - `--no-check-memory` to disable checks on both `gk` and `lmm` commands
  - `estimate_lmm_memory()` API for programmatic memory estimation
  - 50% safety margin based on empirical JAX overhead benchmarks
- **RSS memory logging**: Track memory usage at workflow boundaries
- **Incremental result writing**: Results written per-SNP/per-chunk to disk
  - `output_path` parameter in `run_lmm_association()`
  - JAX streaming runner writes per-file-chunk
- **Safe chunk size defaults**: `MAX_SAFE_CHUNK=50,000` prevents int32 overflow
- **Test tier system**: `tier0` (fast), `tier1` (parity), `tier2` (scale) markers

### Changed

- Memory now bounded by chunk size, not total SNP count
- CLI lmm command uses incremental writing by default
- Eigendecomposition uses numpy LAPACK (not scipy) for large matrix support

### Removed

- Rust/faer eigendecomposition backend (unreliable at scale, higher memory overhead)
- Multi-backend infrastructure (Backend type, `JAMMA_BACKEND` env var, `-be` CLI flag)

### Fixed

- Pre-flight memory check now accounts for full pipeline peak (eigendecomp), not just kinship
- Pre-flight check accounts for SNP count in non-streaming path (JAX genotype copy)
- Eigendecomposition memory check prevents OOM

## [1.0.0] - 2026-02-01

### Added

- **Kinship matrix computation** (`-gk 1`): Centered relatedness matrix XX'/p
- **LMM Wald test** (`-lmm 1`): Univariate linear mixed model association
- **Pre-computed kinship input** (`-k`): Load kinship from file
- **PLINK binary format**: `.bed/.bim/.fam` file support
- **Streaming I/O**: Handle 200k+ samples without loading full matrix
- **JAX acceleration**: CPU/GPU support via JAX backend
- **GEMMA-compatible output**: Identical `.assoc.txt` and `.cXX.txt` formats
- **Numerical equivalence**: Results match GEMMA (identical significance calls, rankings, directions)

### Performance

- 7x faster than GEMMA on kinship computation
- 4x faster than GEMMA on LMM association
- Streaming kinship for datasets exceeding memory

[5.1.4]: https://github.com/michael-denyer/jamma/compare/v5.1.3...v5.1.4
[5.1.3]: https://github.com/michael-denyer/jamma/compare/v5.1.2...v5.1.3
[5.1.2]: https://github.com/michael-denyer/jamma/compare/v5.1.1...v5.1.2
[5.1.1]: https://github.com/michael-denyer/jamma/compare/v5.1.0...v5.1.1
[5.1.0]: https://github.com/michael-denyer/jamma/compare/v5.0.1...v5.1.0
[5.0.1]: https://github.com/michael-denyer/jamma/compare/v5.0.0...v5.0.1
[5.0.0]: https://github.com/michael-denyer/jamma/compare/v4.1.0...v5.0.0
[4.1.0]: https://github.com/michael-denyer/jamma/compare/v4.0.3...v4.1.0
[4.0.3]: https://github.com/michael-denyer/jamma/compare/v4.0.2...v4.0.3
[4.0.2]: https://github.com/michael-denyer/jamma/compare/v4.0.1...v4.0.2
[4.0.1]: https://github.com/michael-denyer/jamma/compare/v4.0.0...v4.0.1
[4.0.0]: https://github.com/michael-denyer/jamma/compare/v3.5.1...v4.0.0
[3.5.1]: https://github.com/michael-denyer/jamma/compare/v3.5.0...v3.5.1
[3.5.0]: https://github.com/michael-denyer/jamma/compare/v3.4.1...v3.5.0
[3.4.1]: https://github.com/michael-denyer/jamma/compare/v3.4.0...v3.4.1
[3.4.0]: https://github.com/michael-denyer/jamma/compare/v3.3.2...v3.4.0
[3.3.2]: https://github.com/michael-denyer/jamma/compare/v3.3.1...v3.3.2
[3.3.1]: https://github.com/michael-denyer/jamma/compare/v3.3.0...v3.3.1
[3.3.0]: https://github.com/michael-denyer/jamma/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/michael-denyer/jamma/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/michael-denyer/jamma/compare/v3.0.1...v3.1.0
[3.0.1]: https://github.com/michael-denyer/jamma/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/michael-denyer/jamma/compare/v2.12.0...v3.0.0
[2.12.0]: https://github.com/michael-denyer/jamma/compare/v2.11.2...v2.12.0
[2.11.2]: https://github.com/michael-denyer/jamma/compare/v2.11.1...v2.11.2
[2.11.1]: https://github.com/michael-denyer/jamma/compare/v2.11.0...v2.11.1
[2.11.0]: https://github.com/michael-denyer/jamma/compare/v2.10.1...v2.11.0
[2.10.1]: https://github.com/michael-denyer/jamma/compare/v2.10.0...v2.10.1
[2.10.0]: https://github.com/michael-denyer/jamma/compare/v2.9.6...v2.10.0
[2.9.6]: https://github.com/michael-denyer/jamma/compare/v2.9.5...v2.9.6
[2.9.5]: https://github.com/michael-denyer/jamma/compare/v2.9.4...v2.9.5
[2.9.4]: https://github.com/michael-denyer/jamma/compare/v2.9.3...v2.9.4
[2.9.3]: https://github.com/michael-denyer/jamma/compare/v2.9.2...v2.9.3
[2.9.2]: https://github.com/michael-denyer/jamma/compare/v2.9.1...v2.9.2
[2.9.1]: https://github.com/michael-denyer/jamma/compare/v2.9.0...v2.9.1
[2.9.0]: https://github.com/michael-denyer/jamma/compare/v2.8.3...v2.9.0
[2.8.3]: https://github.com/michael-denyer/jamma/compare/v2.8.2...v2.8.3
[2.8.2]: https://github.com/michael-denyer/jamma/compare/v2.8.1...v2.8.2
[2.8.1]: https://github.com/michael-denyer/jamma/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/michael-denyer/jamma/compare/v2.7.1...v2.8.0
[2.7.1]: https://github.com/michael-denyer/jamma/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/michael-denyer/jamma/compare/v2.6.1...v2.7.0
[2.6.1]: https://github.com/michael-denyer/jamma/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/michael-denyer/jamma/compare/v2.5.8...v2.6.0
[2.5.8]: https://github.com/michael-denyer/jamma/compare/v2.5.7...v2.5.8
[2.5.7]: https://github.com/michael-denyer/jamma/compare/v2.5.6...v2.5.7
[2.5.6]: https://github.com/michael-denyer/jamma/compare/v2.5.5...v2.5.6
[2.5.5]: https://github.com/michael-denyer/jamma/compare/v2.5.4...v2.5.5
[2.5.4]: https://github.com/michael-denyer/jamma/compare/v2.5.3...v2.5.4
[2.5.3]: https://github.com/michael-denyer/jamma/compare/v2.5.2...v2.5.3
[2.5.2]: https://github.com/michael-denyer/jamma/compare/v2.5.1...v2.5.2
[2.5.1]: https://github.com/michael-denyer/jamma/compare/v2.5.0...v2.5.1
[2.5.0]: https://github.com/michael-denyer/jamma/compare/v2.4.5...v2.5.0
[2.4.5]: https://github.com/michael-denyer/jamma/compare/v2.4.4...v2.4.5
[2.4.4]: https://github.com/michael-denyer/jamma/compare/v2.4.3...v2.4.4
[2.4.3]: https://github.com/michael-denyer/jamma/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/michael-denyer/jamma/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/michael-denyer/jamma/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/michael-denyer/jamma/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/michael-denyer/jamma/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/michael-denyer/jamma/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/michael-denyer/jamma/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/michael-denyer/jamma/compare/v1.5.1...v2.0.0
[1.5.1]: https://github.com/michael-denyer/jamma/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/michael-denyer/jamma/compare/v1.4.3...v1.5.0
[1.4.3]: https://github.com/michael-denyer/jamma/compare/v1.4.2...v1.4.3
[1.3.0]: https://github.com/michael-denyer/jamma/compare/v1.2.0...v1.3
[1.2.0]: https://github.com/michael-denyer/jamma/compare/v0.3.2...v1.2.0
[1.1.0]: https://github.com/michael-denyer/jamma/releases/tag/v1.2.0
[1.0.0]: https://github.com/michael-denyer/jamma/releases/tag/v1.2.0
