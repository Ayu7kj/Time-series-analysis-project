"""Generate consistent LaTeX result tables from the saved analysis outputs."""
import json

import pandas as pd

from analysis import ROOT, TARGETS


def table(name, headers, rows):
    header = " & ".join(r"\Th{" + str(x) + "}" for x in headers)
    lines = [header, *(" & ".join(str(x) for x in row) for row in rows)]
    return "\\def\\"+name+"{\\RptTable{%\n"+"\n".join(x+"\\cr" for x in lines)+"\n}}\n"


def score(x, key):
    return f"{x:,.1f}" if key == "peak_mw" else f"{x:,.2f}"


def p_value(p):
    if p >= .001:
        return f"{p:.4f}"
    a, b = f"{p:.2e}".split("e")
    return f"\u0024{a}\\times 10^{{{int(b)}}}\u0024"


def make_tables(results=ROOT/"results"):
    summary = json.loads((results/"summary.json").read_text(encoding="utf-8"))
    comp = pd.read_csv(results/"model_comparison.csv")
    adf = pd.read_csv(results/"adf.csv")
    granger = pd.read_csv(results/"granger.csv")
    residual = pd.read_csv(results/"residual_summary.csv")
    seasons = pd.read_csv(results/"seasonal_errors.csv")
    extremes = pd.read_csv(results/"largest_test_errors.csv")
    audit = summary["audit"]
    keys = list(TARGETS)
    fields = ["Peak power", "Daily energy", "Mean temperature"]
    defs = []
    defs.append(table("AuditTable", ["Tamil Nadu electricity field", "Peak (MW)", "Energy (MU)"],
                      [(label,*(f"{audit[k][col]:,}" for k in keys)) for label,col in [
                       ("Source rows","rows"), ("Distinct source dates","n_unique"),
                       ("Extra rows on duplicate dates","duplicate_rows"),
                       ("Dates with disagreeing values","conflicting_dates"),
                       ("Blank or nonpositive values in source","nonpositive_or_missing"),
                       ("Missing grid days in common calendar","missing_in_calendar"),
                       ("Longest missing run (days)","longest_gap")]]))
    defs.append(table("SplitTable", ["Partition", "First date", "Last date", "Calendar days", "Observed each"],
                      [(k.title(), v["start"], v["end"], f"{v['n']:,}",
                        f"{v['observed']['peak_mw']:,}") for k,v in summary["splits"].items()]))
    d = summary["describe"]
    defs.append(table("DescriptiveTable", ["Series", "Mean", "SD", "Minimum", "Maximum"],
                      [(label,*(score(d[k][c],k) for c in ("mean","sd","min","max")))
                       for k,label in zip([*keys,"temp_mean_c"],fields)]))
    defs.append(table("CorrelationTable", ["Temperature paired with", "Daily levels", "Daily changes, training"],
                      [(label, *(f"{summary['correlations'][k][c]:.3f}" for c in
                                  ("pearson_level","pearson_change_train"))) for k,label in zip(keys,fields)]))
    defs.append(table("AdfTable", ["Series", "Transformation", "Lags", "Usable rows", "ADF", "5\\% decision"],
                      [(dict(zip([*keys,"temp_mean_c"],fields))[v.series], v.transformation, v.lag,
                        f"{v.n:,}",f"{v.statistic:.2f}","Reject" if v.reject else "Retain")
                       for v in adf.itertuples()]))
    defs.append(table("ResidualTable", ["Target", "Model", "Mean", "SD", "Lag 1", "Lag 7", "Lag 14"],
                      [(("Peak" if v.target==keys[0] else "Energy"),v.model,
                        score(v.mean,v.target),score(v.sd,v.target),f"{v.acf_1:.3f}",f"{v.acf_7:.3f}",f"{v.acf_14:.3f}")
                       for v in residual.itertuples()]))
    defs.append(table("GrangerTable", ["Direction", "Lags", "F statistic", "Classical p", "HAC p", "Holm HAC p"],
                      [(("Temp. to " if v.direction.startswith("temperature") else "")+
                        ("peak" if v.target==keys[0] else "energy")+
                        ("" if v.direction.startswith("temperature") else " to temp."),
                        v.df1,f"{v.f:.2f}",p_value(v.p),p_value(v.p_hac),p_value(v.holm_hac))
                       for v in granger.itertuples()]))
    temp=summary["temperature"]
    defs.append(table("WeatherTable", ["Temperature forecast", "Validation MAE", "Validation RMSE", "Test MAE", "Test RMSE"],
                      [("Fourier + AR(7)",*(f"{temp[s][m]:.3f}" for s in ("validation","test") for m in ("mae","rmse"))),
                       ("Last observed temperature","", "", *(f"{temp['test_persistence'][m]:.3f}" for m in ("mae","rmse")))]))
    for key, label in zip(keys, ("PeakResults","EnergyResults")):
        rows=[]
        for row in comp[comp.target==key].itertuples():
            fn = (lambda x: "\\Th{"+score(x,key)+"}") if row.model == "SARIMAX" else (lambda x: score(x,key))
            spec = row.spec.replace(" + temperature",r" + $T$")
            rows.append((row.model, spec, score(row.val_mae,key),score(row.val_rmse,key),
                         fn(row.test_mae),fn(row.test_rmse)))
        defs.append(table(label, ["Model", "Selected specification", "Val. MAE", "Val. RMSE", "Test MAE", "Test RMSE"],rows))
    rows=[]
    for key,label in zip(keys,fields):
        data = comp[comp.target==key].set_index("model")
        val=summary["paired_sarimax_minus_sarima"][key]
        improvement=100*(1-data.loc["SARIMAX","test_rmse"]/data.loc["SARIMA","test_rmse"])
        low,high=(score(val[x],key) for x in ("bootstrap_lower","bootstrap_upper"))
        rows.append((label,score(val["difference"],key),f"[{low}, {high}]",f"{improvement:.2f}\\%"))
    defs.append(table("PairedTable", ["Target", "RMSE change", "95\\% block interval", "RMSE reduction"], rows))
    rows=[]
    for season in ("Winter","Summer","SW monsoon","NE monsoon"):
        a=seasons[(seasons.model=="SARIMAX")&(seasons.season==season)].set_index("target")
        rows.append((season,int(a.loc[keys[0],"n"]),*(score(a.loc[k,c],k) for k in keys for c in ("mae","rmse"))))
    defs.append(table("SeasonTable", ["Season", "Days", "Peak MAE", "Peak RMSE", "Energy MAE", "Energy RMSE"],rows))
    rows=[]
    for key,label in zip(keys,fields):
        for row in extremes[extremes.target==key].head(2).itertuples():
            rows.append((label,row.date,row.horizon,score(row.actual,key),score(row.forecast,key),score(row.error,key)))
    defs.append(table("ExtremeTable", ["Target", "Date", "Lead", "Actual", "Forecast", "Error"],rows))
    (ROOT/"tables.tex").write_text(
        "% Generated by report_tables.py. Values come from the reproducible analysis.\n"+"\n".join(defs),
        encoding="utf-8")


if __name__=="__main__":
    make_tables()
