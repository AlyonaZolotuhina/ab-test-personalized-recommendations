import numpy as np
from scipy import stats

def var_ratio(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.mean(y) == 0:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    return vx/my**2 + vy*mx**2/my**4 - 2*mx*cov/my**3

def mde_absolute(std, n_per_group, alpha=0.05, power=0.80):
    if not np.isfinite(std) or n_per_group <= 0:
        return np.nan
    return (stats.norm.ppf(1-alpha/2) + stats.norm.ppf(power)) * std * np.sqrt(2/n_per_group)

def bootstrap_diff(control, test, n_boot=5000, seed=42):
    control, test = np.asarray(control, float), np.asarray(test, float)
    control, test = control[np.isfinite(control)], test[np.isfinite(test)]
    if len(control) < 2 or len(test) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    diffs = np.array([rng.choice(test, len(test), replace=True).mean()
                      - rng.choice(control, len(control), replace=True).mean()
                      for _ in range(n_boot)])
    return test.mean()-control.mean(), np.quantile(diffs, .025), np.quantile(diffs, .975)
