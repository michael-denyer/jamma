"""Write association reference values from the independent dense oracle."""

import csv

import numpy as np

from tests.math_validation.fixtures import EXTERNAL_HEADERS


def write_oracle_assoc(model, destination, mode=1, *, metadata=None):
    """Write one oracle association file from explicit model arrays."""
    from tests.math_validation.dense_oracle import all_test_statistics, optimize

    k = np.asarray(model["kinship"])
    x = np.asarray(model["genotypes"])
    y = np.asarray(model["phenotype"])
    w = np.asarray(model.get("covariates", np.ones((len(y), 1))))
    snp_ids = model.get("selected_snp_ids", model.get("snp_ids"))
    rows, details = [], []
    for j, snp in enumerate(snp_ids):
        fit = optimize(k, w, x[:, j], y)
        if mode != 1:
            fit.update(all_test_statistics(k, w, x[:, j], y))
        if mode == 3:
            fit["beta"], fit["se"] = fit["score_beta"], fit["score_se"]
        details.append({"rs": snp, **fit})
        row = (
            metadata(j, snp)
            if metadata is not None
            else {
                "chr": 1,
                "rs": snp,
                "ps": 100 + j,
                "n_miss": model.get("selected_n_miss", [0] * len(snp_ids))[j],
                "allele1": "A",
                "allele0": "G",
                "af": model.get("selected_af", np.mean(x, axis=0))[j] / 2,
            }
        )
        row.update({field: fit[field] for field in EXTERNAL_HEADERS[mode][7:]})
        rows.append(row)
    with destination.open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=EXTERNAL_HEADERS[mode], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    return details
