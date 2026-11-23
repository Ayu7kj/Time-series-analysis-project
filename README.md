# Seasonal Time Series Forecasting with Exogenous Weather Models

*A leak-free week-ahead benchmark on 11+ years of daily grid demand (2013–2024), built from scratch in NumPy and C*

📄 **[Read the Full Report (PDF)](./TSA_report.pdf)** · 📓 **[Explore the Notebook](./timeseries_forecasting.ipynb)**

---

## Overview

This project benchmarks five classical time series models for week-ahead (7-day block) forecasting of **daily peak power met (MW)** and **daily energy met (MU)** in Tamil Nadu across 11+ years of daily records (1 January 2013 – 26 April 2024).

All statistical estimators, optimizers, and hypothesis tests—including conditional Gaussian likelihood ARIMA/SARIMAX, Nelder–Mead optimization, damped Holt–Winters, BIC-selected ADF tests, and Newey–West HAC Granger causality—are implemented **from scratch in pure NumPy**, with optional C acceleration (`recursions.c`) for the inner filtering loops.

To avoid look-ahead leakage, future temperature is never taken from the test set. Instead, a separate out-of-sample weather model (**annual Fourier mean + AR(7) on departures**) generates 7-day temperature forecasts at each origin to supply operational exogenous inputs to SARIMAX.

---

## Key Findings

* **SARIMAX achieves the lowest held-out error on both targets**, reducing test RMSE relative to a weekly seasonal-naive baseline by **17.0%** for peak power (`951.3 MW`) and **27.5%** for daily energy (`19.13 MU`).
* **Weekly seasonality and differencing drive most of the gain.** Standalone `ARIMA(1,0,1)` collapses out of sample (`1,490.1 MW` and `36.56 MU` RMSE) due to mean reversion as demand climbs to record highs, while weekly SARIMA and additive damped Holt–Winters track both the upward trend and the sharp Sunday dip.
* **Strong in-sample temperature signal shrinks out of sample.** Lagged temperature changes strongly predict demand changes in training (HAC-robust, Holm-adjusted Granger tests: $p = 1.49 \times 10^{-6}$ for peak, $p = 5.52 \times 10^{-12}$ for energy). Out of sample, adding forecasted temperature reduces RMSE by **1.32%** (peak) and **2.02%** (energy) over matched SARIMA models, with 28-day circular moving-block bootstrap 95% intervals spanning zero (`[-33.8, 5.6] MW` and `[-0.95, 0.15] MU`).
* **Tail errors are highly concentrated.** A single unexplained shock week (1–7 May 2023) accounts for **25.1%** of SARIMAX's total squared peak error and **17.2%** of its squared energy error across 618 test days.

---

## Test Set Performance (618 Held-Out Days)

Evaluated across **89 rolling 7-day forecast blocks** (15 August 2022 – 26 April 2024) with fixed refitted parameters and zero look-ahead:

| Model | Peak Spec | Peak MAE (MW) | Peak RMSE (MW) | Energy Spec | Energy MAE (MU) | Energy RMSE (MU) |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **Seasonal Naive** | Weekly lag 7 | 856.2 | 1,146.6 | Weekly lag 7 | 20.21 | 26.40 |
| **ARIMA** | `(1,0,1)` | 1,297.2 | 1,490.1 | `(1,0,1)` | 32.10 | 36.56 |
| **SARIMA** | `(0,1,1)(0,1,1)[7]` | 705.4 | 964.1 | `(1,1,1)(0,1,1)[7]` | 14.81 | 19.53 |
| **Holt-Winters** | Additive, damped | 690.3 | 952.9 | Additive, damped | 14.58 | 20.39 |
| **SARIMAX** | `(0,1,1)(0,1,1)[7] + T` | **683.7** | **951.3** | `(1,1,1)(0,1,1)[7] + T` | **14.25** | **19.13** |

*`T` denotes daily mean temperature supplied by the separate Fourier + AR(7) weather forecaster (`0.745 °C` test RMSE vs. `0.820 °C` persistence).*

---

## Methodology & Engineering Highlights

* **From-Scratch Statistical Engine (`models.py` + `recursions.c`):** Implemented ARIMA/SARIMAX conditional least-squares, reflection-parameterized stationarity/invertibility constraints, Nelder–Mead optimization, additive damped Holt–Winters, ADF, ACF/PACF, and bivariate VAR HAC Granger tests without `statsmodels` or `scipy`. Inner ARMA innovation and Holt–Winters loops are accelerated via a C shared library loaded with `ctypes`.
* **Calendar-Safe Data Hygiene:** Audited 3,955 source rows per target (resolving 55 duplicate rows and 7 conflicting dates via within-date medians). Retained all 236 missing calendar days as explicit grid gaps so lag-7 always stays aligned with the same weekday, applying a strictly causal historical fill only inside forecasting recursions.
* **Leak-Free Model Selection:** Chronological 70% / 15% / 15% split (Train / Validation / Test). Candidate ARIMA/SARIMA orders are shortlisted by AIC/BIC strictly within fixed $(d, D)$ pairs and selected across differencing scales using validation block RMSE.
* **Inference & Uncertainty:** Predictive direction tested via Newey–West HAC Granger causality with Holm multiple-testing adjustment; matched SARIMAX vs. SARIMA RMSE differences evaluated via 3,000-replicate paired circular moving-block bootstraps (28-day blocks).

---

## Repository Structure

```text
├── TSA_report.pdf                  # Full 19-page technical report
├── timeseries_forecasting.ipynb    # Interactive notebook with cached result tables
├── analysis.py                     # End-to-end data audit, model selection, and rolling evaluation
├── models.py                       # From-scratch ARIMA/SARIMAX, Holt-Winters, ADF, and Granger tests
├── recursions.c                    # Optional C kernels for ARMA innovations and Holt-Winters filtering
├── figures.py                      # Vector-PDF chart engine built on ReportLab
├── report_tables.py                # Generates formatted LaTeX summary tables
├── requirements.txt                # Python dependencies
├── data/                           # Source Tamil Nadu electricity and weather extracts
├── results/                        # Saved forecasts, diagnostics, and metric tables
└── figures/                        # All 15 vector-PDF figures used in the report
