"""Explore a fully invested, long-only maximum-Sharpe allocation (rf = 0)."""

import argparse
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SERIES = ["NASDAQCOM_PC1", "NASDAQNQCAN_PC1", "NASDAQNQGBN_PC1", "NASDAQNQJPN_PC1"]
PROJECT_DIR = Path(__file__).resolve().parent


def load_returns(data_dir):
    """Align the four percentage series on dates and convert to decimals."""
    tables = []
    for name in SERIES:
        path = data_dir / f"{name}.xlsx"
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}. See data/README.md.")
        table = pd.read_excel(path, engine="openpyxl")
        required = {"observation_date", name}
        if not required.issubset(table.columns):
            raise ValueError(f"{path.name} needs columns {sorted(required)}")
        table = table[["observation_date", name]].copy()
        table["observation_date"] = pd.to_datetime(table["observation_date"], errors="raise")
        if table["observation_date"].isna().any() or table["observation_date"].duplicated().any():
            raise ValueError(f"{path.name} contains missing or duplicate dates")
        table[name] = pd.to_numeric(table[name], errors="raise")
        tables.append(table)
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on="observation_date", how="inner", validate="one_to_one")
    returns = merged.set_index("observation_date").sort_index()[SERIES] / 100
    if len(returns) < 2 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Need at least two common dates with finite values in all four series")
    return returns


def negative_sharpe(weights, mean_returns, covariance):
    """Minimizing negative return/volatility maximizes Sharpe when rf = 0."""
    volatility = np.sqrt(weights @ covariance @ weights)
    if volatility <= 0:
        return 1e10
    return -(weights @ mean_returns) / volatility


def analyze(returns, output_dir):
    mean_returns = returns.mean()
    covariance = returns.cov()
    for title, values in [
        ("PER-SERIES MEAN", mean_returns),
        ("PER-SERIES STANDARD DEVIATION", returns.std(ddof=1)),
        ("PER-SERIES VARIANCE", returns.var(ddof=1)),
        ("COVARIANCE MATRIX", covariance),
    ]:
        print(f"\n===== {title} (per observation) =====")
        print(values.to_string(float_format=lambda value: f"{value:.6f}"))

    mean_vector = mean_returns.to_numpy()
    covariance_matrix = covariance.to_numpy()
    number_of_series = len(SERIES)
    result = minimize(
        negative_sharpe,
        np.ones(number_of_series) / number_of_series,
        args=(mean_vector, covariance_matrix),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * number_of_series,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}],
    )
    if not result.success:
        raise RuntimeError(f"Optimization did not converge: {result.message}")
    weights = result.x
    if not np.isclose(weights.sum(), 1, atol=1e-6) or (weights < -1e-6).any():
        raise RuntimeError("Optimizer returned an infeasible allocation")
    portfolio_return = float(weights @ mean_vector)
    portfolio_variance = float(weights @ covariance_matrix @ weights)
    portfolio_volatility = np.sqrt(portfolio_variance)
    if portfolio_volatility <= 0:
        raise ValueError("Portfolio volatility must be positive to calculate Sharpe")
    print(f"\nAligned observations: {len(returns)}; {returns.index.min().date()} to {returns.index.max().date()}")
    print("\n===== MAX SHARPE WEIGHTS (long-only, rf = 0) =====")
    print(pd.Series(weights, index=SERIES).to_string(float_format=lambda value: f"{value:.6f}"))
    print("\n===== PORTFOLIO STATISTICS (per observation) =====")
    print(f"Mean return: {portfolio_return:.6f}")
    print(f"Standard deviation: {portfolio_volatility:.6f}")
    print(f"Variance: {portfolio_variance:.6f}")
    print(f"Sharpe ratio: {portfolio_return / portfolio_volatility:.6f}")
    # Preserve the original arithmetic annualization; validity depends on inputs.
    print("\n===== CONDITIONAL ANNUALIZATION (assuming monthly returns) =====")
    print("Only meaningful if inputs are actual monthly returns, not year-over-year changes.")
    print(f"Annualized mean return: {portfolio_return * 12:.6f}")
    print(f"Annualized standard deviation: {portfolio_volatility * np.sqrt(12):.6f}")
    print(f"Annualized Sharpe ratio: {portfolio_return / portfolio_volatility * np.sqrt(12):.6f}")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar([name.removesuffix("_PC1") for name in SERIES], weights * 100, color="#26689a")
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set(ylabel="Portfolio weight (%)", ylim=(0, 110), title="Maximum-Sharpe allocation | Long-only, rf = 0")
    figure.tight_layout()
    figure.savefig(output_dir / "portfolio_weights.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    args = parser.parse_args()
    returns = load_returns(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "portfolio_results.txt"
    with report.open("w", encoding="utf-8") as stream, redirect_stdout(stream):
        analyze(returns, args.output_dir)
    print(f"Saved report: {report}")
    print(f"Saved chart: {args.output_dir / 'portfolio_weights.png'}")


if __name__ == "__main__":
    main()
