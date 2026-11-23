"""Render all project charts as vector PDFs directly from the result tables."""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

from analysis import ROOT, MODEL_NAMES, SEASONS, TARGETS

BLUE, GREEN, ORANGE, PURPLE, GRAY = "#245789", "#13796e", "#b7502b", "#754a91", "#607382"
INK, GRID, LIGHT = "#173147", "#dce6ec", "#f1f6f8"
MODEL_COLORS = [GRAY, BLUE, GREEN, PURPLE, ORANGE]


def ink(col):
    return HexColor(col)


def label(c, x, y, s, size=9, color=INK, align="left", bold=False):
    c.setFillColor(ink(color))
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    {"left": c.drawString, "right": c.drawRightString, "center": c.drawCentredString}[align](x,y,str(s))


def ticks(lo, hi, count=5):
    width = (hi-lo)/count
    unit = 10**np.floor(np.log10(width))
    step = next(x*unit for x in (1,2,2.5,5,10) if x*unit >= width)
    return np.arange(np.ceil(lo/step)*step, hi+0.001*step, step)


def formatted(v):
    return f"{v:,.1f}".rstrip("0").rstrip(".") if 0 < abs(v) < 10 and v != int(v) else f"{v:,.0f}"


class Axes:
    def __init__(self, c, box, xs, ys, xt, yt=None):
        self.c = c
        self.left, self.bottom, self.w, self.h = box
        self.xs, self.ys = xs, ys
        for v in (ticks(*ys) if yt is None else yt):
            y = self.y(v)
            c.setStrokeColor(ink(GRID))
            c.setLineWidth(.55)
            c.line(self.left,y,self.left+self.w,y)
            label(c,self.left-6,y-3,formatted(v),8,GRAY,"right")
        for v, s in xt:
            label(c,self.x(v), self.bottom-14, s, 8, GRAY, "center")
        c.setStrokeColor(ink(GRID))
        c.line(self.left,self.bottom,self.left+self.w,self.bottom)

    def x(self, v):
        return self.left + (v-self.xs[0])/(self.xs[1]-self.xs[0])*self.w

    def y(self, v):
        return self.bottom + (v-self.ys[0])/(self.ys[1]-self.ys[0])*self.h

    def line(self, x, y, col, weight=1.2, dash=False):
        c = self.c
        c.setStrokeColor(ink(col))
        c.setLineWidth(weight)
        c.setDash(3,2) if dash else c.setDash()
        c.setLineJoin(1)
        p = c.beginPath()
        prev = False
        for a,b in zip(x,y):
            if np.isfinite(b):
                (p.lineTo if prev else p.moveTo)(self.x(a),self.y(b))
                prev = True
            else:
                prev = False
        c.drawPath(p)
        c.setDash()

    def dots(self, x, y, col, size=2, alpha=1.):
        self.c.setFillColor(ink(col))
        self.c.setFillAlpha(alpha)
        for a,b in zip(x,y):
            if np.isfinite(a) and np.isfinite(b):
                self.c.circle(self.x(a),self.y(b),size,stroke=0,fill=1)
        self.c.setFillAlpha(1)

    def bars(self, x, y, col, width=.6, base=0.):
        for i,(xx, yy) in enumerate(zip(x,y)):
            if np.isfinite(yy):
                self.c.setFillColor(ink(col if isinstance(col,str) else col[i]))
                lo, hi = sorted([self.y(base), self.y(yy)])
                self.c.rect(self.x(xx-width/2),lo,self.x(xx+width/2)-self.x(xx-width/2),hi-lo,stroke=0,fill=1)

    def boxplot(self, x, y, color=GREEN):
        p10,q1,median,q3,p90 = np.nanpercentile(y,[10,25,50,75,90])
        c = self.c
        here = self.x(x)
        c.setStrokeColor(ink(color))
        c.setFillColor(ink(LIGHT))
        c.setLineWidth(.95)
        c.line(here,self.y(p10),here,self.y(p90))
        c.rect(here-12,self.y(q1),24,self.y(q3)-self.y(q1),stroke=1,fill=1)
        c.line(here-12,self.y(median),here+12,self.y(median))


def limits(y, pad=.06, floor_zero=False):
    lo,hi = np.nanmin(y),np.nanmax(y)
    return (0. if floor_zero else lo-pad*(hi-lo), hi+pad*(hi-lo))


def date_x(dates, freq):
    elapsed = (dates-dates[0]).days.to_numpy()
    axis = pd.date_range(dates[0], dates[-1], freq=freq)
    labels = [(int((d-dates[0]).days), d.strftime("%Y" if freq=="YS" else "%b %y")) for d in axis]
    return elapsed, labels


def legend(c, entries, x, y, spacing=129):
    for col,s in entries:
        c.setStrokeColor(ink(col))
        c.setLineWidth(2)
        c.line(x,y+3,x+14,y+3)
        label(c,x+19,y,s,8)
        x += spacing


def chart(path, height, draw):
    c = Canvas(str(path), pagesize=(510,height), pageCompression=1)
    c.setCreator("Tamil Nadu electricity forecasting project")
    draw(c)
    c.showPage()
    c.save()


def draw_figures(results=ROOT/"results", out=ROOT/"figures"):
    out.mkdir(parents=True,exist_ok=True)
    df = pd.read_csv(results/"tamil_nadu_clean_daily.csv", parse_dates=["date"]).set_index("date")
    corr = pd.read_csv(results/"demand_correlograms.csv", keep_default_na=False)
    comp = pd.read_csv(results/"model_comparison.csv")
    test = pd.read_csv(results/"forecasts_test.csv",parse_dates=["date"])
    season = pd.read_csv(results/"seasonal_errors.csv")
    horizon = pd.read_csv(results/"horizon_errors.csv")
    weather = pd.read_csv(results/"temperature_test.csv",parse_dates=["date"]).set_index("date")
    x, years = date_x(df.index,"YS")

    def save(number, slug, h, fn):
        chart(out/f"{number:02d}_{slug}.pdf",h,fn)

    def demand_history(c):
        for i,(key,(heading,unit)) in enumerate(TARGETS.items()):
            bottom = 168-138*i
            y = df[key]
            label(c,61,bottom+108,f"{heading} ({unit})",10,bold=True)
            ax=Axes(c,(61,bottom,422,100),(0,x[-1]),limits(y.to_numpy()),years)
            ax.line(x,y,BLUE,.45)
            ax.line(x,y.rolling(90,min_periods=60).mean(),ORANGE,1.5)
        legend(c,[(BLUE,"Observed daily"),(ORANGE,"Trailing 90-day mean")],61,3)
    save(1,"demand_history",292,demand_history)

    def annual(c):
        for i,(key,(_,unit)) in enumerate(TARGETS.items()):
            left=49+i*252
            grouped=df[key].groupby(df.index.year).mean()
            v=np.arange(len(grouped))
            label(c,left,169,"Peak power (MW)" if i==0 else "Energy (MU)",10,bold=True)
            ax=Axes(c,(left,31,187,125),(-.6,11.6),limits(grouped.to_numpy(),.11),[(j,str(k)[2:]) for j,k in enumerate(grouped.index)])
            ax.line(v[:-1],grouped.iloc[:-1],BLUE,1.5)
            ax.line(v[-2:],grouped.iloc[-2:],GRAY,1.1,True)
            ax.dots(v[:-1],grouped.iloc[:-1],BLUE,2.3)
            ax.dots(v[-1:],grouped.iloc[-1:],ORANGE,3)
        label(c,49,4,"Orange markers show partial-year 2024 averages.",8,GRAY)
    save(2,"annual_demand",181,annual)

    def monthly(c):
        for i,(key,(_,unit)) in enumerate(TARGETS.items()):
            bottom=111-88*i
            m=df[key].groupby(df.index.month).mean()
            label(c,66,bottom+69,"Peak power (MW)" if i==0 else "Energy (MU)",9,bold=True)
            ax=Axes(c,(66,bottom,417,60),(.5,12.5),limits(m.to_numpy(),.12),
                    [(k,calendar.month_abbr[k]) for k in range(1,13)] if i else [])
            ax.line(m.index,m,GREEN,1.5)
            ax.dots(m.index,m,GREEN,2.8)
    save(3,"monthly_demand",193,monthly)

    def weekday(c):
        for i,(key,(_,unit)) in enumerate(TARGETS.items()):
            left=48+i*251
            weekday=(df[key]-df[key].groupby(df.index.to_period("M")).transform("mean")).groupby(df.index.weekday).mean()
            label(c,left,153,"Peak departure (MW)" if i==0 else "Energy departure (MU)",9,bold=True)
            ax=Axes(c,(left,29,187,112),(-.6,6.6),limits(weekday.to_numpy(),.17),
                    [(k,calendar.day_abbr[k]) for k in range(7)])
            ax.bars(range(7),weekday,[GREEN]*5+[ORANGE]*2,.62)
    save(4,"weekday_demand",170,weekday)

    def by_season(c):
        for i,(key,(_,unit)) in enumerate(TARGETS.items()):
            left=49+i*250
            data=[df.loc[df.season==s,key].dropna() for s in SEASONS]
            label(c,left,157,"Peak power (MW)" if i==0 else "Energy (MU)",9,bold=True)
            ax=Axes(c,(left,30,187,115),(-.55,3.55),limits(np.concatenate([np.percentile(v,[10,90]) for v in data]),.12),
                    [(j,k) for j,k in enumerate(["Winter","Summer","SW","NE"])])
            for j,v in enumerate(data):
                ax.boxplot(j,v,ORANGE if j==1 else GREEN)
        label(c,49,5,"Boxes show the middle 50%; whiskers show the 10th and 90th percentiles.",8,GRAY)
    save(5,"seasonal_demand",172,by_season)

    def temperature(c):
        ax=Axes(c,(52,31,425,137),(0,x[-1]),limits(df[["temp_min_c","temp_max_c"]].to_numpy()),years)
        for col,key in [(ORANGE,"temp_max_c"),(GREEN,"temp_mean_c"),(BLUE,"temp_min_c")]:
            ax.line(x,df[key].rolling(30,min_periods=28).mean(),col,1.05)
        label(c,52,177,"Temperature (degrees C), trailing 30-day means",10,bold=True)
        legend(c,[(ORANGE,"Daily maximum"),(GREEN,"Daily mean"),(BLUE,"Daily minimum")],55,5)
    save(6,"temperature_history",193,temperature)

    def scatter(c):
        for i,(key,(_,unit)) in enumerate(TARGETS.items()):
            left=51+i*253
            data=df[[key,"temp_mean_c"]].dropna()
            limits_x=limits(data.temp_mean_c.to_numpy(),.035)
            label(c,left,194,"Peak power (MW)" if i==0 else "Energy (MU)",9,bold=True)
            ax=Axes(c,(left,39,186,142),limits_x,limits(data[key].to_numpy()),
                    [(k,str(k)) for k in range(21,33,2) if limits_x[0]<k<limits_x[1]])
            ax.dots(data.temp_mean_c,data[key],BLUE,1,.18)
            group=data.groupby(pd.cut(data.temp_mean_c,np.arange(20,35,.75)),observed=True).mean()
            ax.line(group.temp_mean_c,group[key],ORANGE,1.75)
            ax.dots(group.temp_mean_c,group[key],ORANGE,2)
            label(c,left+93,12,"Daily mean temperature (degrees C)",8,GRAY,"center")
    save(7,"temperature_scatter",207,scatter)

    for i,(key,(field,_)) in enumerate(TARGETS.items()):
        def correlogram(c,key=key):
            for row,tr in enumerate(("Level","First difference")):
                for j,col in enumerate(("acf","pacf")):
                    x0=47+j*249
                    y0=149-row*117
                    s=corr[(corr.series==key)&(corr.transformation==tr)&(corr.lag>0)]
                    ax=Axes(c,(x0,y0,192,83),(.2,35.8),(-.48,1.03),[(k,k) for k in (7,14,21,28,35)],[-.4,0,.4,.8])
                    ax.bars(s.lag,s[col],BLUE if row==0 else GREEN,.65)
                    bound=1.96/np.sqrt(s.n.iloc[0])
                    for value in (-bound,bound):
                        ax.line([.2,35.8],[value,value],ORANGE,.55,True)
                    label(c,x0,y0+91,tr+"  /  "+col.upper(),9,bold=True)
                    label(c,x0+192,y0-26,"lag (days)",8,GRAY,"right")
        save(8+i,"peak_correlograms" if i==0 else "energy_correlograms",252,correlogram)

    def weather_score(c):
        date=pd.DatetimeIndex(weather.index)
        s,xlabels=date_x(date,"2MS")
        ax=Axes(c,(52,32,423,154),(0,s[-1]),limits(weather[["actual_c","predicted_c"]].to_numpy()),xlabels)
        ax.line(s,weather.actual_c,INK,.9)
        ax.line(s,weather.predicted_c,ORANGE,.85)
        legend(c,[(INK,"Observed temperature"),(ORANGE,"Seven-day block forecast")],53,8)
        label(c,52,195,"Daily mean temperature (degrees C)",10,bold=True)
    save(10,"weather_forecast",209,weather_score)

    for i,(key,(field,unit)) in enumerate(TARGETS.items()):
        def forecast(c,key=key,field=field,unit=unit):
            scores=comp[comp.target==key].set_index("model")
            winner=scores.test_rmse.idxmin()
            data=test[test.target==key].set_index("date")
            s,xlabels=date_x(data.index,"2MS")
            ax=Axes(c,(57,36,424,175),(0,s[-1]),limits(data[["actual",winner]].to_numpy()),xlabels)
            ax.line(s,data.actual,INK,1.02)
            ax.line(s,data[winner],ORANGE,1.0)
            label(c,57,222,f"{field} ({unit})",10,bold=True)
            legend(c,[(INK,"Actual"),(ORANGE,winner)],58,8)
        save(11+i,"peak_forecast" if i==0 else "energy_forecast",237,forecast)

    def season_error(c):
        for i,(key,(_,unit)) in enumerate(TARGETS.items()):
            left=46+i*251
            data=season[season.target==key]
            winner=comp[comp.target==key].sort_values("test_rmse").iloc[0].model
            scores=data[data.model==winner].set_index("season").reindex(SEASONS)
            ax=Axes(c,(left,36,193,119),(-.5,3.5),limits(scores.mae.to_numpy(),.17,True),
                    [(j,k) for j,k in enumerate(["Winter","Summer","SW","NE"])])
            ax.bars(range(4),scores.mae,[GREEN,ORANGE,GREEN,GREEN],.62)
            for j,val in enumerate(scores.mae):
                label(c,ax.x(j),ax.y(val)+4,f"{val:.1f}",8,GRAY,"center")
            label(c,left,165,f"{'Peak' if i==0 else 'Energy'} MAE ({unit})",9,bold=True)
    save(14,"season_forecast_errors",183,season_error)

    def horizon_error(c):
        for i,(key,(_,unit)) in enumerate(TARGETS.items()):
            left=48+i*251
            data=horizon[horizon.target==key]
            ax=Axes(c,(left,36,190,120),(.6,7.4),limits(data.rmse.to_numpy(),.12),[(h,h) for h in range(1,8)])
            for model,col in zip(MODEL_NAMES,MODEL_COLORS):
                v=data[data.model==model]
                ax.line(v.horizon,v.rmse,col,1.15)
                ax.dots(v.horizon,v.rmse,col,1.7)
            label(c,left,167,f"{'Peak' if i==0 else 'Energy'} RMSE ({unit})",9,bold=True)
        legend(c,list(zip(MODEL_COLORS,MODEL_NAMES)),48,4,93)
    save(15,"horizon_forecast_errors",181,horizon_error)

    def residuals(c):
        for i,(key,(_,unit)) in enumerate(TARGETS.items()):
            left=50+i*251
            scores=comp[comp.target==key].set_index("model")
            winner=scores.test_rmse.idxmin()
            dm=test[test.target==key].set_index("date")
            error=dm[winner]-dm.actual
            vals,edges=np.histogram(error.dropna(),bins=25)
            pos=(edges[1:]+edges[:-1])/2
            xmarks=[(v,formatted(v)) for v in ticks(edges[0],edges[-1],4)]
            ax=Axes(c,(left,36,187,115),(edges[0],edges[-1]),(0,vals.max()*1.12),xmarks)
            ax.bars(pos,vals,GREEN,(edges[1]-edges[0])*.9)
            ax.line([0,0],[0,vals.max()*1.04],ORANGE,1.1,True)
            label(c,left,164,f"{'Peak' if i==0 else 'Energy'} error ({unit})",9,bold=True)
        label(c,50,7,"Test-block forecast minus actual; vertical reference at zero.",8,GRAY)
    save(13,"heldout_error_distribution",180,residuals)
    print(f"Rendered 15 vector charts to {out}")


if __name__=="__main__":
    draw_figures()
