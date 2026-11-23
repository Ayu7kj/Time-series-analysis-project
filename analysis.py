"""Read the Tamil Nadu extracts and run the complete forecast experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models import Arima, HoltWinters, SeasonalNaive, TempForecast, acf, adf, granger, pacf

ROOT = Path(__file__).resolve().parent
TARGETS = {"peak_mw": ("Peak power met", "MW"), "energy_mu": ("Daily energy met", "MU")}
MODEL_NAMES = ("Seasonal naive", "ARIMA", "SARIMA", "Holt-Winters", "SARIMAX")
SEASONS = ("Winter", "Summer", "SW monsoon", "NE monsoon")
SEASON_MONTH = ["Winter", "Winter", "Summer", "Summer", "Summer", "SW monsoon",
                "SW monsoon", "SW monsoon", "SW monsoon", "NE monsoon", "NE monsoon", "NE monsoon"]
WEATHER = {"2m_temperature_min": "temp_min_c", "2m_temperature_mean": "temp_mean_c",
           "2m_temperature_max": "temp_max_c", "2m_dewpoint_temperature_mean": "dewpoint_mean_c",
           "surface_solar_radiation_downwards_mean": "solar_radiation_mean_native",
           "total_cloud_cover_mean": "cloud_cover_mean"}


def input_file(folder, names):
    for name in names:
        path = folder / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Expected one of {names} in {folder}")


def load_data(folder):
    specifications = [
        ("peak_mw", ("corrected_peak_met_MW.csv", "peak_tn_extract.csv")),
        ("energy_mu", ("corrected_daily_energy_met_MU (1).csv", "corrected_daily_energy_met_MU.csv", "energy_tn_extract.csv")),
    ]
    electricity, audit = [], {}
    for name, filenames in specifications:
        d = pd.read_csv(input_file(folder, filenames), usecols=["Date", "Tamil Nadu"])
        d["Date"] = pd.to_datetime(d["Date"], format="%Y-%m-%d")
        d["Tamil Nadu"] = pd.to_numeric(d["Tamil Nadu"], errors="coerce").where(lambda s: s > 0)
        grouped = d.groupby("Date")["Tamil Nadu"]
        series = grouped.median().rename(name)
        audit[name] = {"rows": len(d), "start": str(d.Date.min().date()),
                       "end": str(d.Date.max().date()), "n_unique": len(series),
                       "duplicate_rows": len(d)-len(series), "conflicting_dates": int(grouped.nunique().gt(1).sum()),
                       "nonpositive_or_missing": int(d["Tamil Nadu"].isna().sum())}
        electricity.extend([series, grouped.size().rename(f"{name}_source_rows"),
                            grouped.nunique().gt(1).rename(f"{name}_conflict")])
    weather_file = input_file(folder, ("Tamil_Nadu (2).csv", "Tamil_Nadu.csv", "weather_tn_extract.csv"))
    w = pd.read_csv(weather_file, usecols=["date", *WEATHER])
    w["date"] = pd.to_datetime(w["date"], format="%Y-%m-%d")
    if w.date.duplicated().any():
        raise ValueError("Weather keys must be unique.")
    w = w.set_index("date").rename(columns=WEATHER).sort_index()
    for field in ("temp_min_c", "temp_mean_c", "temp_max_c", "dewpoint_mean_c"):
        w[field] = pd.to_numeric(w[field], errors="coerce") - 273.15
    complete_t = w[["temp_min_c", "temp_mean_c", "temp_max_c"]].dropna()
    if not ((complete_t.temp_min_c <= complete_t.temp_mean_c) &
            (complete_t.temp_mean_c <= complete_t.temp_max_c)).all():
        raise ValueError("Daily temperature ordering is inconsistent.")
    start = max(pd.Timestamp(audit[x]["start"]) for x in TARGETS)
    first_weather, last_weather = w.temp_mean_c.first_valid_index(), w.temp_mean_c.last_valid_index()
    start = max(start, first_weather)
    end = min(last_weather, *(pd.Timestamp(audit[x]["end"]) for x in TARGETS))
    calendar = pd.date_range(start, end, freq="D", name="date")
    frame = pd.concat(electricity, axis=1).reindex(calendar).join(w)
    for name in TARGETS:
        frame[f"{name}_source_rows"] = frame[f"{name}_source_rows"].fillna(0).astype(int)
        frame[f"{name}_conflict"] = frame[f"{name}_conflict"].eq(True)
        missing = frame[name].isna().to_numpy()
        stretches = np.diff(np.flatnonzero(np.r_[True, ~missing, True]))
        audit[name]["missing_in_calendar"] = int(missing.sum())
        audit[name]["longest_gap"] = int(stretches.max() - 1)
    frame["weekday"] = calendar.day_name()
    frame["season"] = np.array(SEASON_MONTH)[calendar.month - 1]
    audit["weather"] = {"extract_rows": len(w), "extract_start": str(w.index.min().date()),
                        "extract_end": str(w.index.max().date()), "last_mean": str(last_weather.date()),
                        "missing_in_calendar": int(frame.temp_mean_c.isna().sum()),
                        "missing_columns": {x: int(frame[x].isna().sum()) for x in WEATHER.values()}}
    audit["calendar"] = {"start": str(start.date()), "end": str(end.date()), "rows": len(frame),
                         "complete_core": int(frame[["peak_mw", "energy_mu", "temp_mean_c"]].dropna().shape[0])}
    return frame, audit


def causal_fill(y):
    """Use the most recent actual same-weekday value within 28 days, else last actual."""
    values = y.copy()
    weekday_last = np.full(7, -1000)
    previous = None
    for i, value in enumerate(y):
        weekday = i % 7
        if np.isfinite(value):
            previous = value
            weekday_last[weekday] = i
        elif previous is None:
            raise ValueError("The modeling calendar must start with an observation.")
        else:
            j = weekday_last[weekday]
            values[i] = y[j] if i-j <= 28 else previous
    return values


def metrics(actual, prediction):
    error = (prediction-actual)[np.isfinite(actual) & np.isfinite(prediction)]
    return {"n": len(error), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(np.mean(error))}


def innovation_summary(errors):
    valid = np.isfinite(errors)
    e = errors[valid]
    result = {"n": len(e), "mean": float(e.mean()), "sd": float(e.std(ddof=1))}
    for lag in (1, 7, 14):
        paired = valid[lag:] & valid[:-lag]
        result[f"acf_{lag}"] = float(np.corrcoef(errors[lag:][paired], errors[:-lag][paired])[0,1])
    return result


def paired_uncertainty(actual, baseline, augmented, seed):
    """Paired circular moving-block bootstrap of the test RMSE difference."""
    e1, e2 = (baseline-actual)**2, (augmented-actual)**2
    rng = np.random.default_rng(seed)
    block = 28
    index = (rng.integers(len(actual), size=(3000, int(np.ceil(len(actual)/block)), 1)) +
             np.arange(block)[None,None,:]) % len(actual)
    sample = index.reshape(len(index), -1)[:,:len(actual)]
    delta = np.sqrt(np.nanmean(e2[sample],axis=1)) - np.sqrt(np.nanmean(e1[sample],axis=1))
    lo, hi = np.quantile(delta, [.025,.975])
    return {"difference": float(np.sqrt(np.nanmean(e2))-np.sqrt(np.nanmean(e1))),
            "bootstrap_lower": float(lo), "bootstrap_upper": float(hi),
            "replications": len(index), "block_days": block}


def rolling_temperature(model, day, temperature, first, last):
    predictions = np.full(last-first, np.nan)
    for origin in range(first, last, 7):
        h = min(7, last-origin)
        predictions[origin-first:origin-first+h] = model.forecast(day[:origin], temperature[:origin], h)
    return predictions


def rolling_forecast(model, values, temperature, predicted_temperature, first, last):
    predictions = np.full(last-first, np.nan)
    for origin in range(first, last, 7):
        h = min(7, last-origin)
        begin = origin-first
        predictions[begin:begin+h] = model.forecast(values[:origin], h, temperature[:origin],
                                                  predicted_temperature[begin:begin+h])
    return predictions


def name_order(order):
    p, d, q, sp, sd, sq = order
    regular = f"({p},{d},{q})"
    return f"{regular}({sp},{sd},{sq})[7]" if sp+sd+sq else regular


def candidate_orders(seasonal):
    regular = [(0,1,0), (1,1,0), (0,1,1), (1,1,1), (2,1,1), (1,1,2), (1,0,1), (2,0,1)]
    if not seasonal:
        return [o + (0,0,0) for o in regular]
    return [o + s for o in [(0,1,1), (1,1,0), (1,1,1), (1,0,1)]
            for s in [(1,0,0), (0,0,1), (1,0,1), (0,1,1), (1,1,0)]]


def model_fit(y, raw, t, val_weather, a, b, candidates, target, label, rows):
    results = []
    for order in candidates:
        model = Arima(order).fit(y[:a], np.isfinite(raw[:a]))
        pred = rolling_forecast(model, y, t, val_weather, a, b)
        score = metrics(raw[a:b], pred)
        row = {"target": target, "family": label, "spec": name_order(order), "aic": model.aic,
               "bic": model.bic, "n_fit": model.n, "converged": model.converged, **score}
        rows.append(row)
        results.append((order, row))
    # Screen within comparable differencing groups, then rank by validation RMSE.
    eligible = []
    for d, ds in sorted({(o[1], o[4]) for o, _ in results}):
        group = [(o, r) for o, r in results if (o[1], o[4]) == (d, ds) and r["converged"]]
        for key in ("aic", "bic"):
            eligible.extend(sorted(group, key=lambda z: z[1][key])[:3])
    return min(eligible, key=lambda z: (z[1]["rmse"], z[1]["mae"]))[0]


def holm(p):
    p = np.array(p)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def describe(frame):
    stats = {}
    for field in (*TARGETS, "temp_mean_c"):
        s = frame[field]
        month = s.groupby(s.index.month).mean()
        year = s.groupby(s.index.year).agg(["mean", "count"])
        day = s.groupby(s.index.weekday).mean()
        change = s.diff()
        median = change.median()
        mad = (change-median).abs().median()
        flagged = (change-median).abs() > 6*1.4826*mad
        dates = (change-median).abs().nlargest(5).index
        stats[field] = {"mean": float(s.mean()), "sd": float(s.std()), "min": float(s.min()),
                        "max": float(s.max()), "max_date": str(s.idxmax().date()),
                        "highest_month": int(month.idxmax()), "lowest_month": int(month.idxmin()),
                        "month_means": {str(k): float(v) for k, v in month.items()},
                        "year_means": {str(k): float(v) for k, v in year["mean"].items()},
                        "year_counts": {str(k): int(v) for k, v in year["count"].items()},
                        "weekday_means": day.tolist(), "n_flags": int(flagged.sum()),
                        "largest_changes": [{"date": str(i.date()), "change": float(change[i])} for i in dates]}
    return stats


def diagnostics(frame, filled, train_end):
    records, lag_plot = [], []
    for key in (*TARGETS, "temp_mean_c"):
        raw = frame[key].iloc[:train_end].to_numpy()
        for kind, s in (("Level", raw), ("First difference", np.diff(raw)),
                        ("Weekly difference", raw[7:]-raw[:-7]),
                        ("First & weekly", np.diff(raw[7:]-raw[:-7]))):
            if key == "temp_mean_c" and kind not in ("Level", "First difference"):
                continue
            records.append({"series": key, "transformation": kind, **adf(s)})
        if key not in TARGETS:
            continue
        for kind, s in (("Level", filled[key][:train_end]),
                        ("First difference", np.diff(filled[key][:train_end]))):
            for lag, (r, p) in enumerate(zip(acf(s, 35), pacf(s, 35))):
                lag_plot.append({"series": key, "transformation": kind, "lag": lag,
                                 "acf": float(r), "pacf": float(p), "n": len(s)})
    return pd.DataFrame(records), pd.DataFrame(lag_plot)


def run(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    frame, audit = load_data(folder)
    n = len(frame)
    a, b = int(.70*n), int(.85*n)
    frame["split"] = np.repeat(["train", "validation", "test"], [a, b-a, n-b])
    frame.to_csv(out/"tamil_nadu_clean_daily.csv", float_format="%.8g")
    day = (frame.index - pd.Timestamp("2000-01-01")).days.to_numpy()
    filled = {k: causal_fill(frame[k].to_numpy()) for k in (*TARGETS, "temp_mean_c")}
    t = filled["temp_mean_c"]
    t_raw = frame.temp_mean_c.to_numpy()
    tmask = np.isfinite(t_raw)
    adf_table, lag_table = diagnostics(frame, filled, a)
    adf_table.to_csv(out/"adf.csv", index=False)
    lag_table.to_csv(out/"demand_correlograms.csv", index=False)
    temperature_grid = []
    for order in (1, 2, 3, 7):
        model = TempForecast(order).fit(day[:a], t[:a], tmask[:a])
        pred = rolling_temperature(model, day, t, a, b)
        temperature_grid.append((order, metrics(t_raw[a:b], pred), pred))
    t_order, t_validation_score, val_weather = min(temperature_grid, key=lambda z: z[1]["rmse"])
    t_model = TempForecast(t_order).fit(day[:b], t[:b], tmask[:b])
    test_weather = rolling_temperature(t_model, day, t, b, n)
    t_validation = pd.DataFrame({"date": frame.index[a:b], "actual_c": t_raw[a:b],
                                 "predicted_c": val_weather, "horizon": np.arange(b-a)%7+1})
    t_testing = pd.DataFrame({"date": frame.index[b:], "actual_c": t_raw[b:], "predicted_c": test_weather,
                              "horizon": np.arange(n-b)%7+1})
    t_validation.to_csv(out/"temperature_validation.csv", index=False)
    t_testing.to_csv(out/"temperature_test.csv", index=False)

    comparisons, predictions, grid, causal, seasonal, horizon, extremes, residuals = [], [], [], [], [], [], [], []
    differences = {}
    for key in TARGETS:
        raw, y = frame[key].to_numpy(), filled[key]
        causal.extend([{"target": key, **row} for row in granger(raw[:a], t_raw[:a], day[:a],
                                                              frame.index.weekday.to_numpy()[:a])])
        order_a = model_fit(y, raw, t, val_weather, a, b, candidate_orders(False), key, "ARIMA", grid)
        order_s = model_fit(y, raw, t, val_weather, a, b, candidate_orders(True), key, "SARIMA", grid)
        hw_grid = []
        for damped in (False, True):
            model = HoltWinters(damped).fit(y[:a], np.isfinite(raw[:a]))
            response = rolling_forecast(model, y, t, val_weather, a, b)
            result = metrics(raw[a:b], response)
            hw_grid.append((result["rmse"], damped))
            grid.append({"target": key, "family": "Holt-Winters", "spec": "damped" if damped else "undamped",
                         "converged": model.converged, **result})
        damped = min(hw_grid)[1]
        factory = [("Weekly lag 7", SeasonalNaive), (name_order(order_a), lambda: Arima(order_a)),
                   (name_order(order_s), lambda: Arima(order_s)),
                   ("Additive, damped" if damped else "Additive, undamped", lambda: HoltWinters(damped)),
                   (name_order(order_s)+" + temperature", lambda: Arima(order_s, weather=True))]
        wide = pd.DataFrame({"date": frame.index[b:], "target": key, "actual": raw[b:], "horizon": np.arange(n-b)%7+1,
                             "season": frame.season.iloc[b:].to_numpy()})
        for label, (spec, make) in zip(MODEL_NAMES, factory):
            initial, final = make(), make()
            validation, testing = np.array([]), np.array([])
            for model, lo, hi, tf in [(initial, a, b, val_weather), (final, b, n, test_weather)]:
                if isinstance(model, Arima):
                    model.fit(y[:lo], np.isfinite(raw[:lo]), t[:lo])
                elif isinstance(model, HoltWinters):
                    model.fit(y[:lo], np.isfinite(raw[:lo]))
                predicted = rolling_forecast(model, y, t, tf, lo, hi)
                if lo == a:
                    validation = predicted
                else:
                    testing = predicted
            score_val, score_test = metrics(raw[a:b], validation), metrics(raw[b:], testing)
            comparison = {"target": key, "model": label, "spec": spec,
                          "converged_train": getattr(initial, "converged", True),
                          "converged_refit": getattr(final, "converged", True),
                          **{f"val_{k}": v for k,v in score_val.items()},
                          **{f"test_{k}": v for k,v in score_test.items()}}
            if isinstance(final, Arima):
                comparison.update(bic=final.bic, aic=final.aic)
                if final.weather:
                    comparison.update(beta=final.beta)
            elif isinstance(final, HoltWinters):
                comparison.update(alpha=final.params[0], smoothing_beta=final.params[1],
                                  gamma=final.params[2], phi=final.params[3])
            comparisons.append(comparison)
            errors = final.residuals(y[:b], np.isfinite(raw[:b]), t[:b])
            residuals.append({"target": key, "model": label, **innovation_summary(errors)})
            wide[label] = testing
            for season in SEASONS:
                mask = wide.season.eq(season).to_numpy()
                seasonal.append({"target": key, "model": label, "season": season,
                                 **metrics(raw[b:][mask], testing[mask])})
            for h in range(1, 8):
                mask = wide.horizon.eq(h).to_numpy()
                horizon.append({"target": key, "model": label, "horizon": h,
                                **metrics(raw[b:][mask], testing[mask])})
        champion = min((row for row in comparisons if row["target"] == key), key=lambda row: row["test_rmse"])["model"]
        differences[key] = paired_uncertainty(raw[b:], wide.SARIMA.to_numpy(), wide.SARIMAX.to_numpy(),
                                              928 + list(TARGETS).index(key))
        for i in (wide[champion]-wide.actual).abs().nlargest(5).index:
            extremes.append({"target": key, "model": champion, "date": str(wide.date[i].date()),
                             "actual": float(wide.actual[i]), "forecast": float(wide[champion][i]),
                             "error": float(wide[champion][i]-wide.actual[i]), "horizon": int(wide.horizon[i])})
        predictions.append(wide)
    causal = pd.DataFrame(causal)
    causal["holm_hac"] = holm(causal.p_hac)
    seasonal, horizon = pd.DataFrame(seasonal), pd.DataFrame(horizon)
    comparisons, predictions = pd.DataFrame(comparisons), pd.concat(predictions, ignore_index=True)
    for name, table in {"model_comparison": comparisons, "forecasts_test": predictions,
                        "candidate_grid": pd.DataFrame(grid), "granger": causal,
                        "seasonal_errors": seasonal, "horizon_errors": horizon,
                        "largest_test_errors": pd.DataFrame(extremes), "residual_summary": pd.DataFrame(residuals)}.items():
        table.to_csv(out/f"{name}.csv", index=False)
    correlations = {}
    for key in TARGETS:
        pair = frame[[key, "temp_mean_c"]]
        correlations[key] = {"pearson_level": float(pair.corr().iloc[0,1]),
                             "pearson_change_train": float(pair.iloc[:a].diff().corr().iloc[0,1])}
    summary = {"audit": audit, "describe": describe(frame), "correlations": correlations,
               "paired_sarimax_minus_sarima": differences,
               "splits": {str(k): {"n": len(x), "start": str(x.index[0].date()),
                                   "end": str(x.index[-1].date()),
                                   "observed": {field: int(x[field].notna().sum()) for field in TARGETS}}
                          for k, x in frame.groupby("split", sort=False)},
               "temperature": {"lag": t_order, "validation": t_validation_score,
                               "test": metrics(t_raw[b:], test_weather),
                               "test_persistence": metrics(t_raw[b:], np.repeat(t[np.arange(b-1,n-1,7)],7)[:n-b]),
                               "ar": t_model.ar.tolist(), "grid": [(p, s) for p,s,_ in temperature_grid]}}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(comparisons[["target","model","spec","val_rmse","test_mae","test_rmse","converged_refit"]].to_string(index=False))
    print(causal.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"data")
    parser.add_argument("--out", type=Path, default=ROOT/"results")
    options = parser.parse_args()
    run(options.data, options.out)
