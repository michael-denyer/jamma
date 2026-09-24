"""GCTA ``--info`` oracle: a literal transliteration of Geno.cpp in Python ints.

Source: ``jianyangqt/gcta`` ``src/Geno.cpp`` at commit c6bbbee,
``Geno::getGenoDouble_bgen`` and ``calDosage_bgen``. Only the autosomal,
unphased path is transliterated: JAMMA's decoder rejects phased data, and the
chrX male adjustment (lines 1280 to 1311) does not touch the INFO sums. The
integer sums are Python ints, so they cannot overflow; every floating-point
step is a Python float (IEEE double) in the order GCTA writes it.
"""

from __future__ import annotations

from collections.abc import Iterable


def gcta_info(
    q11: Iterable[int],
    q12: Iterable[int],
    missing: Iterable[bool],
    bits_prob: int,
    keep: Iterable[int],
) -> float:
    """Return GCTA's INFO for one variant over the kept samples ``keep``.

    Args:
        q11: Stored P(11) numerator per sample, in file order.
        q12: Stored P(12) numerator per sample, in file order.
        missing: The ploidy byte's missingness bit per sample.
        bits_prob: The variant's bit depth B.
        keep: ``sampleKeepIndex``: kept sample positions, in order.

    Returns:
        INFO as GCTA computes it. With no non-missing kept sample GCTA
        divides 0.0 by 0.0, and this returns the NaN that C would give.
    """
    q11, q12, missing = list(q11), list(q12), list(missing)
    # 1225: uint64_t mask = (1U << bits_prob) - 1;
    mask = (1 << bits_prob) - 1
    # 1226: uint64_t dosage_sum = 0, fij_sum = 0, dosage2_sum = 0;
    dosage_sum = 0
    fij_sum = 0
    dosage2_sum = 0
    valid_n = 0
    valid_allele = 0
    # 1242: for(int j = 0; j < curSampleCT; j++){ sindex = sampleKeepIndex[j];
    for sindex in keep:
        # 1245: if(item_ploidy > 128){ miss_index.push_back(sindex); ... }
        if missing[sindex]:
            continue
        # 1254-1255: prob1 = geno_temp & mask; prob2 = (geno_temp >> B) & mask;
        prob1 = int(q11[sindex])
        prob2 = int(q12[sindex])
        # 1097-1100 calDosage_bgen: prob1d = prob1 * 2; dosage = prob1d + prob2;
        prob1d = prob1 * 2
        dosage = prob1d + prob2
        # 1264-1271
        dosage_sum += dosage
        dosage2_sum += dosage * dosage
        fij_sum += prob1d
        valid_n += 1
        valid_allele += 2

    # 1278: double dosage_sum_half = dosage_sum;
    dosage_sum_half = float(dosage_sum)
    # 1318: double maskd = (double)mask;
    maskd = float(mask)
    if valid_allele == 0:
        # C: 0.0 / maskd / 0 is NaN, NaN < 1e-50 is false, so info is NaN.
        return float("nan")
    # 1319: double af = (double)dosage_sum_half / maskd / validAllele;
    af = dosage_sum_half / maskd / valid_allele
    # 1325: double std = 2.0 * af * (1.0 - af);
    std = 2.0 * af * (1.0 - af)
    # 1327: double mask2 = mask * mask;  (uint64 product, then double)
    mask2 = float(mask * mask)
    # 1328-1333
    if std < 1e-50:
        info = 1.0
    else:
        dos2_fij_sum = float(dosage_sum + fij_sum) / maskd - float(dosage2_sum) / mask2
        info = 1.0 - dos2_fij_sum / (std * valid_n)
    return info
