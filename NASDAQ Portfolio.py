import os
import pandas as pd
import numpy as np

print("Current working directory:", os.getcwd())

# === 1. Set your file paths ===
# Use absolute paths 
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
        # force engine for .xlsx; requires `pip install openpyxl`
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        print(f"!! ERROR reading {path}: {e}")
        raise

    # Expect columns ["observation_date", <name>]
    if "observation_date" not in df.columns:
        raise ValueError(f"'observation_date' column not found in {path}. Columns: {df.columns}")
    if name not in df.columns:
        raise ValueError(f"Expected column '{name}' not found in {path}. Columns: {df.columns}")

    data[name] = df[["observation_date", name]]

print("\nAll files loaded successfully.")

# === 3. Merge on observation_date ===
merged = data["NASDAQCOM_PC1"]
for key in ["NASDAQNQCAN_PC1", "NASDAQNQGBN_PC1", "NASDAQNQJPN_PC1"]:
    merged = merged.merge(data[key], on="observation_date")

merged["observation_date"] = pd.to_datetime(merged["observation_date"])
merged = merged.set_index("observation_date")

# === 4. Treat columns as return series ===
returns = merged[["NASDAQCOM_PC1", "NASDAQNQCAN_PC1", "NASDAQNQGBN_PC1", "NASDAQNQJPN_PC1"]].copy()

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

# === 6. Max Sharpe ratio portfolio (rf = 0, shorting allowed) ===
mu    = mean_returns.values
Sigma = cov_matrix.values

Sigma_inv = np.linalg.inv(Sigma)
w_unnorm  = Sigma_inv @ mu
weights   = w_unnorm / w_unnorm.sum()

asset_names    = returns.columns
weights_series = pd.Series(weights, index=asset_names, name="Weight")

port_mean = float(weights @ mu)
port_var  = float(weights @ Sigma @ weights)
port_std  = np.sqrt(port_var)
sharpe    = port_mean / port_std

print("\n===== MAX SHARPE PORTFOLIO WEIGHTS (sum to 1, shorting allowed) =====")
print(weights_series)

print("\n===== PORTFOLIO STATISTICS (monthly) =====")
print(f"Expected return: {port_mean:.6f}")
print(f"Standard deviation: {port_std:.6f}")
print(f"Variance: {port_var:.6f}")
print(f"Sharpe ratio (rf=0): {sharpe:.6f}")

# Optional: annualize assuming 12 months/year
annual_mean   = port_mean * 12
annual_std    = port_std * np.sqrt(12)
annual_sharpe = annual_mean / annual_std

print("\n===== ANNUALIZED PORTFOLIO STATISTICS (assuming 12 months/year) =====")
print(f"Annualized expected return: {annual_mean:.6f}")
print(f"Annualized standard deviation: {annual_std:.6f}")
print(f"Annualized Sharpe ratio (rf=0): {annual_sharpe:.6f}")
