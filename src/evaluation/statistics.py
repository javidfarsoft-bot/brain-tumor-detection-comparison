"""Statistical significance tests comparing Faster R-CNN vs. YOLO (Phase 8)."""
import numpy as np
from scipy import stats


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10000, alpha: float = 0.05):
    values = np.asarray(values)
    means = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_resamples)
    ])
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(values)), (float(lower), float(upper))


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000):
    a, b = np.asarray(a), np.asarray(b)
    observed_diff = np.mean(a - b)
    diffs = a - b
    count = 0
    for _ in range(n_resamples):
        signs = np.random.choice([1, -1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    p_value = count / n_resamples
    return observed_diff, p_value


def wilcoxon_test(a: np.ndarray, b: np.ndarray):
    stat, p = stats.wilcoxon(a, b)
    return stat, p


def mcnemar_test(b: int, c: int):
    from statsmodels.stats.contingency_tables import mcnemar
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=(b + c < 25))
    return result.statistic, result.pvalue


def cohens_d(a: np.ndarray, b: np.ndarray):
    a, b = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
