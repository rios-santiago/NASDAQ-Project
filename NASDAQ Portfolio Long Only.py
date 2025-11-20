# nasdaq_portfolio_long_only.py

import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize

print("Current working directory:", os.getcwd())

# === 1. Set your file paths ===
# Use absolute paths or put the files in this folder and just use filenames.
# Here I show absolute paths with FORWARD SLASHES to avoid Windows escape issues.

files = {
    "NASDAQCOM_PC1": r"C:\Users\santi_nuaavil\Downloads\NASDAQCOM_PC1.xlsx",
    "NASDAQNQCAN_PC1": r"C:\Users\santi_nuaavil\Downloads\NASDAQNQCAN_PC1.xlsx",
    "NASDAQNQGBN_PC1": r"C:\Users\santi_nuaavil\Downloads\NASDAQNQGBN_PC1.xlsx",
    "NASDAQNQJPN_PC1": r"C:\Users\santi_nuaavil\Downloads\NASDAQNQJPN_PC1.xlsx",
}

# === 2. Load the Excel files safely ===
data = {}

for name, path in files.items():
    print(f"\n--- Loading {name} from: {path}")
    if not os.path.isfile(path):
        print(f"!! ERROR: File not found: {path}")
        raise FileNotFoundError(
            f"File for {name} not found. Check the path or move the .xlsx into {os.getcwd()}"
        )

    try:
        # force engine for .xlsx
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        print(f"!! ERROR reading {path}: {e}")
        raise

    # Expect columns ["observation_date", <name>]
    if "observation_date" not in df.columns:
        raise ValueError(
            f"'observation_date' column not found in {path}. Columns: {df.columns}"
        )
    if name not in df.columns:
        raise ValueError(
            f"Expected column '{name}' not found in {path}. Columns: {df.columns}"
        )

    data[name] = df[["observation_date", name]]

print("\nAll files loaded successfully.")

# === 3. Merge on observation_date ===
merged = data["NASDAQCOM_PC1"]
for key in ["NASDAQNQCAN_PC1", "NASDAQNQGBN_PC1", "NASDAQNQJPN_PC1"]:
    merged = merged.merge(data[key], on="observation_date")

merged["observation_date"] = pd.to_datetime(merged["observation_date"])
merged = merged.set_index("observation_date")

# === 4. Treat columns as monthly return series ===
returns = merged[["NASDAQCOM_PC1",
                  "NASDAQNQCAN_PC1",
                  "NASDAQNQGBN_PC1",
                  "NASDAQNQJPN_PC1"]].copy()

# === 5. Descriptive statistics ===
mean_returns = returns.mean()
std_returns  = returns.std(ddof=1)
var_returns  = returns.var(ddof=1)
cov_matrix   = returns.cov()

print("\n===== PER-ASSET MEAN RETURNS (monthly) =====")
print(mean_returns)

print("\n===== PER-ASSET STANDARD DEVIATIONS (monthly) =====")
print(std_returns)

print("\n===== PER-ASSET VARIANCES (monthly) =====")
print(var_returns)

print("\n===== COVARIANCE MATRIX (monthly) =====")
print(cov_matrix)

# === 6. Max Sharpe ratio portfolio (NO SHORTING, rf = 0) ===

mu    = mean_returns.values
Sigma = cov_matrix.values
n     = len(mu)

def neg_sharpe(weights, mu, Sigma):
    """Negative Sharpe ratio (rf = 0) for scipy minimize."""
    port_return = np.dot(weights, mu)
    port_var    = weights @ Sigma @ weights
    port_std    = np.sqrt(port_var)
    # To avoid division-by-zero issues
    if port_std == 0:
        return 1e10
    return -port_return / port_std

# Constraint: sum of weights = 1
constraints = (
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
)

# Bounds: 0 <= w_i <= 1 (no shorting)
bounds = tuple((0.0, 1.0) for _ in range(n))

# Initial guess: equal weights
initial = np.ones(n) / n

opt_result = minimize(
    neg_sharpe,
    initial,
    args=(mu, Sigma),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
)

if not opt_result.success:
    print("\nOptimization did NOT converge:")
    print(opt_result.message)
else:
    print("\nOptimization converged successfully.")

weights_ns = opt_result.x
weights_ns_series = pd.Series(weights_ns, index=returns.columns, name="Weight (No Shorting)")

# Portfolio statistics (monthly)
port_mean_ns = float(weights_ns @ mu)
port_var_ns  = float(weights_ns @ Sigma @ weights_ns)
port_std_ns  = np.sqrt(port_var_ns)
sharpe_ns    = port_mean_ns / port_std_ns

print("\n===== MAX SHARPE (NO SHORTING) WEIGHTS =====")
print(weights_ns_series)

print("\n===== PORTFOLIO STATS (NO SHORTING, monthly) =====")
print(f"Expected return: {port_mean_ns:.6f}")
print(f"Std deviation:   {port_std_ns:.6f}")
print(f"Variance:        {port_var_ns:.6f}")
print(f"Sharpe ratio:    {sharpe_ns:.6f}")

# === 7. Optional: Annualize assuming 12 months/year ===
annual_mean_ns   = port_mean_ns * 12
annual_std_ns    = port_std_ns * np.sqrt(12)
annual_sharpe_ns = annual_mean_ns / annual_std_ns

print("\n===== ANNUALIZED STATS (NO SHORTING, 12 months/year) =====")
print(f"Annualized expected return: {annual_mean_ns:.6f}")
print(f"Annualized std deviation:   {annual_std_ns:.6f}")
print(f"Annualized Sharpe ratio:    {annual_sharpe_ns:.6f}")
