/**
 * snp_stats.c — Single-pass per-SNP statistics (mean, variance, miss, HWE).
 *
 * Computes mean, population variance, missing count, and optionally HWE
 * genotype counts (n_aa, n_ab, n_bb) per SNP column in a single pass over
 * the data.  Replaces the multi-pass Python/NumPy pattern (separate isnan,
 * nanmean, nanvar, and genotype counting calls).
 *
 * Two entry points: snp_stats_chunk_f32 (float input) and snp_stats_chunk_f64
 * (double input).  Both accumulate in double precision internally.
 *
 * Threading: uses pthreads (not OpenMP) to avoid conflicting with MKL's
 * internal OpenMP on systems where MKL's libiomp5 is loaded.  OpenMP
 * parallel regions in the same .so that dlopen's MKL can corrupt the
 * libiomp5 thread pool (OMP Error #13 at kmp_runtime.cpp).  pthreads
 * sidesteps this entirely.
 *
 * Data layout: (n_samples, n_snps_chunk) C-contiguous row-major.
 * Element (i, j) is at data[i * n_snps_chunk + j].
 */

#include "jlinalg.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- pthread thread pool for snp_stats ---- */

typedef struct {
    const void *data; /* float* or double* */
    npy_intp n_samples;
    npy_intp n_snps_chunk;
    double *means;
    npy_intp *miss_counts;
    double *variances;
    int64_t *n_aa;
    int64_t *n_ab;
    int64_t *n_bb;
    int compute_hwe;
    npy_intp j_start; /* first column for this thread */
    npy_intp j_end;   /* one past last column */
    int is_f32;       /* 1 = float, 0 = double */
} snp_stats_task_t;

static void snp_stats_range(const snp_stats_task_t *t) {
    for (npy_intp j = t->j_start; j < t->j_end; j++) {
        double sum = 0.0;
        double sum_sq = 0.0;
        npy_intp n_miss = 0;
        npy_intp n_valid = 0;
        int64_t cnt_aa = 0, cnt_ab = 0, cnt_bb = 0;

        for (npy_intp i = 0; i < t->n_samples; i++) {
            double dval;
            int is_nan;
            if (t->is_f32) {
                float fval = ((const float *)t->data)[i * t->n_snps_chunk + j];
                is_nan = isnan(fval);
                dval = (double)fval;
                if (!is_nan && t->compute_hwe) {
                    if (fval == 0.0f)
                        cnt_aa++;
                    else if (fval == 1.0f)
                        cnt_ab++;
                    else if (fval == 2.0f)
                        cnt_bb++;
                }
            } else {
                dval = ((const double *)t->data)[i * t->n_snps_chunk + j];
                is_nan = isnan(dval);
                if (!is_nan && t->compute_hwe) {
                    if (dval == 0.0)
                        cnt_aa++;
                    else if (dval == 1.0)
                        cnt_ab++;
                    else if (dval == 2.0)
                        cnt_bb++;
                }
            }
            if (is_nan) {
                n_miss++;
                continue;
            }
            sum += dval;
            sum_sq += dval * dval;
            n_valid++;
        }

        t->miss_counts[j] = n_miss;

        if (n_valid > 0) {
            double mean = sum / (double)n_valid;
            double var = sum_sq / (double)n_valid - mean * mean;
            if (var < 0.0) var = 0.0; /* E[X^2]-E[X]^2 cancellation guard */
            t->means[j] = mean;
            t->variances[j] = var;
        } else {
            t->means[j] = 0.0;
            t->variances[j] = 0.0;
        }

        if (t->compute_hwe) {
            t->n_aa[j] = cnt_aa;
            t->n_ab[j] = cnt_ab;
            t->n_bb[j] = cnt_bb;
        }
    }
}

static void *snp_stats_thread_fn(void *arg) {
    snp_stats_range((const snp_stats_task_t *)arg);
    return NULL;
}

/* ---- Public API (f32 / f64 entry points) ---- */

static void snp_stats_chunk_impl(const void *data, npy_intp n_samples, npy_intp n_snps_chunk,
                                 double *means, npy_intp *miss_counts, double *variances,
                                 int64_t *n_aa, int64_t *n_ab, int64_t *n_bb, int compute_hwe,
                                 int is_f32) {
    int n_threads = jlinalg_get_n_threads();
    if (n_snps_chunk <= 256 || n_threads <= 1) {
        /* Small chunk or single-threaded: run inline */
        snp_stats_task_t task = {
            .data = data,
            .n_samples = n_samples,
            .n_snps_chunk = n_snps_chunk,
            .means = means,
            .miss_counts = miss_counts,
            .variances = variances,
            .n_aa = n_aa,
            .n_ab = n_ab,
            .n_bb = n_bb,
            .compute_hwe = compute_hwe,
            .j_start = 0,
            .j_end = n_snps_chunk,
            .is_f32 = is_f32,
        };
        snp_stats_range(&task);
        return;
    }

    /* Cap threads to column count */
    if (n_threads > n_snps_chunk) n_threads = (int)n_snps_chunk;

    /* Stack-allocate for small thread counts, heap for large */
    snp_stats_task_t tasks_stack[64];
    pthread_t threads_stack[64];
    snp_stats_task_t *tasks =
        n_threads <= 64 ? tasks_stack
                        : (snp_stats_task_t *)malloc((size_t)n_threads * sizeof(snp_stats_task_t));
    pthread_t *threads = n_threads <= 64
                             ? threads_stack
                             : (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));

    if (!tasks || !threads) {
        /* Allocation failed — fall back to single-threaded.
         * This should be rare (< 64 threads uses stack), but if it happens
         * the process is critically low on memory — warn the user. */
        fprintf(stderr,
                "snp_stats: malloc failed for %d threads, "
                "falling back to single-threaded\n",
                n_threads);
        if (tasks != tasks_stack) free(tasks);
        if (threads != threads_stack) free(threads);
        snp_stats_task_t task = {
            .data = data,
            .n_samples = n_samples,
            .n_snps_chunk = n_snps_chunk,
            .means = means,
            .miss_counts = miss_counts,
            .variances = variances,
            .n_aa = n_aa,
            .n_ab = n_ab,
            .n_bb = n_bb,
            .compute_hwe = compute_hwe,
            .j_start = 0,
            .j_end = n_snps_chunk,
            .is_f32 = is_f32,
        };
        snp_stats_range(&task);
        return;
    }

    /* Partition columns across threads (static schedule) */
    npy_intp cols_per_thread = n_snps_chunk / n_threads;
    npy_intp remainder = n_snps_chunk % n_threads;
    npy_intp col = 0;

    for (int t = 0; t < n_threads; t++) {
        npy_intp chunk = cols_per_thread + (t < remainder ? 1 : 0);
        tasks[t] = (snp_stats_task_t){
            .data = data,
            .n_samples = n_samples,
            .n_snps_chunk = n_snps_chunk,
            .means = means,
            .miss_counts = miss_counts,
            .variances = variances,
            .n_aa = n_aa,
            .n_ab = n_ab,
            .n_bb = n_bb,
            .compute_hwe = compute_hwe,
            .j_start = col,
            .j_end = col + chunk,
            .is_f32 = is_f32,
        };
        col += chunk;
    }

    /* Launch threads 1..n-1, run thread 0 inline */
    int n_launched = 1; /* thread 0 runs inline */
    for (int t = 1; t < n_threads; t++) {
        int rc = pthread_create(&threads[t], NULL, snp_stats_thread_fn, &tasks[t]);
        if (rc != 0) {
            fprintf(stderr,
                    "snp_stats: pthread_create failed for thread %d (rc=%d), "
                    "processing remaining %d columns single-threaded\n",
                    t, rc, n_threads - t);
            for (int u = t; u < n_threads; u++)
                snp_stats_range(&tasks[u]);
            n_launched = t;
            break;
        }
        n_launched = t + 1;
    }

    snp_stats_range(&tasks[0]);

    for (int t = 1; t < n_launched; t++)
        pthread_join(threads[t], NULL);

    if (tasks != tasks_stack) free(tasks);
    if (threads != threads_stack) free(threads);
}

void snp_stats_chunk_f32(const float *data, npy_intp n_samples, npy_intp n_snps_chunk,
                         double *means, npy_intp *miss_counts, double *variances, int64_t *n_aa,
                         int64_t *n_ab, int64_t *n_bb, int compute_hwe) {
    snp_stats_chunk_impl(data, n_samples, n_snps_chunk, means, miss_counts, variances, n_aa, n_ab,
                         n_bb, compute_hwe, 1);
}

void snp_stats_chunk_f64(const double *data, npy_intp n_samples, npy_intp n_snps_chunk,
                         double *means, npy_intp *miss_counts, double *variances, int64_t *n_aa,
                         int64_t *n_ab, int64_t *n_bb, int compute_hwe) {
    snp_stats_chunk_impl(data, n_samples, n_snps_chunk, means, miss_counts, variances, n_aa, n_ab,
                         n_bb, compute_hwe, 0);
}
