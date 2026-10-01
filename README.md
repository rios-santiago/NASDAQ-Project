# NASDAQ Portfolio Analysis

A student project exploring how Python can support financial analysis: importing
Excel data, calculating descriptive statistics and covariance, and using SciPy to
explore a long-only portfolio allocation. It demonstrates practical exposure to
Python analysis rather than advanced software development.

## Methodology

The script aligns four NASDAQ-related series on their shared observation dates,
converts percentage values to decimals, and calculates sample means, standard
deviations, variances, and covariance. SciPy's SLSQP optimizer maximizes the ratio
of mean return to volatility, with a zero risk-free rate, weights between 0 and 1,
and total weight equal to 1. It starts from equal weights. The output includes a
text report and one bar chart of the optimized weights.

The original calculation is retained, including conditional arithmetic
annualization: mean × 12, volatility × √12, and Sharpe × √12.

## Inputs and interpretation

The four original Excel workbooks are **not included**. See [data/README.md](data/README.md)
for filenames and column requirements. Their source and sample dates are not
recorded in the original repository.

**Verify the return definition before interpreting performance.** The original
script assumes monthly returns. If `_PC1` denotes percentage change from a year
ago (as in some data exports), monthly observations are year-over-year changes,
not monthly returns. The calculations then describe those changes; the
annualization and portfolio-return interpretation are not valid. No conversion
to monthly returns is attempted here because the underlying data are unavailable.
The square-root annualization also assumes serially uncorrelated returns.

The retained [Portfolio Results.txt](Portfolio%20Results.txt) is an **unverified
historical output**, not a fresh run of this revision. It reports approximately
85.92% NASDAQCOM and 14.08% NASDAQNQCAN. Results depend on the input sample and are
in-sample estimates, not a backtest or evidence of future investment performance.
Index series do not themselves establish investable portfolios; fees and trading
costs are not modeled.

## Run

Python 3.9 or newer:

```bash
python -m pip install -r requirements.txt
# Place the original workbooks in data/, then:
python "NASDAQ Portfolio Long Only.py"
```

The default directories are relative to the script, so it can be launched from
another working directory. Custom locations are also supported:

```bash
python "NASDAQ Portfolio Long Only.py" --data-dir /path/to/workbooks --output-dir /path/to/results
```

Outputs: `outputs/portfolio_results.txt` and `outputs/portfolio_weights.png`.
Input workbooks and generated outputs are excluded from Git by default.

## Validation

The updated workflow was exercised end to end with synthetic Excel inputs,
including chart generation, portability, and missing-input handling. The original
results cannot be reproduced until the original workbooks are supplied. Synthetic
test results are not presented as financial findings.
