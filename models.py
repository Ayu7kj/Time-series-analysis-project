"""Conditional least-squares ARIMA/SARIMA, additive Holt–Winters and tests."""
from __future__ import annotations

import ctypes
from math import erfc, exp, lgamma, log, log1p, sqrt
from pathlib import Path

import numpy as np


def _load_kernels():
    for filename in ("recursions.dll", "recursions.so", "recursions.dylib"):
        path = Path(__file__).with_name(filename)
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
            lib.innovations.argtypes = [array, ctypes.c_int, array, ctypes.c_int,
                                        array, ctypes.c_int, array]
            lib.hw_sse.argtypes = [array, ctypes.c_int, array, array, array]
            lib.innovations.restype = None
            lib.hw_sse.restype = None
            return lib
    return None


_KERNELS = _load_kernels()


def innovations(z, ar, ma):
    """Compute one-step innovations with zero pre-sample data and errors."""
    z, ar, ma = map(lambda a: np.ascontiguousarray(a, dtype=float), (z, ar, ma))
    err = np.empty(len(z))
    if _KERNELS is not None:
        _KERNELS.innovations(z, len(z), ar, len(ar), ma, len(ma), err)
    else:
        adjusted = np.convolve(z, np.r_[1., -ar])[:len(z)]
        for t, value in enumerate(adjusted):
            j = min(t, len(ma))
            err[t] = value - ma[:j] @ err[t-j:t][::-1] if j else value
    return err


def minimize(fun, start, step=.22, maxiter=520, tol=1e-7):
    """Nelder–Mead with convergence in both objective and parameters."""
    start = np.asarray(start, dtype=float)
    k = len(start)
    if k == 0:
        return start, fun(start), True
    vertices = np.vstack([start, start + np.eye(k) * step])
    values = np.array([fun(x) for x in vertices])
    for iteration in range(maxiter):
        order = np.argsort(values)
        vertices, values = vertices[order], values[order]
        if np.max(abs(vertices[1:] - vertices[0])) < 0.002 and values[-1] - values[0] < tol:
            return vertices[0], float(values[0]), True
        center = vertices[:-1].mean(axis=0)
        reflected = 2 * center - vertices[-1]
        fr = fun(reflected)
        if fr < values[0]:
            expanded = center + 2 * (reflected - center)
            fe = fun(expanded)
            vertices[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            vertices[-1], values[-1] = reflected, fr
        else:
            outer = fr < values[-1]
            contracted = (center + (reflected if outer else vertices[-1])) / 2
            fc = fun(contracted)
            if fc < (fr if outer else values[-1]):
                vertices[-1], values[-1] = contracted, fc
            else:
                vertices[1:] = (vertices[0] + vertices[1:]) / 2
                values[1:] = [fun(x) for x in vertices[1:]]
    best = np.argmin(values)
    return vertices[best], float(values[best]), False


def _stationary_coefficients(raw):
    reflection = .985 * np.tanh(raw)
    if len(reflection) == 2:
        reflection[0] *= 1 - reflection[1]
    return reflection


def polynomials(raw, order):
    p, _, q, sp, _, sq = order
    ar = _stationary_coefficients(raw[:p])
    ma = -_stationary_coefficients(raw[p:p+q])
    seasonal_ar = np.r_[1., np.zeros(6), -.985 * np.tanh(raw[p+q])] if sp else [1.]
    seasonal_ma = np.r_[1., np.zeros(6), .985 * np.tanh(raw[p+q+sp])] if sq else [1.]
    return (-np.convolve(np.r_[1., -ar], seasonal_ar)[1:],
            np.convolve(np.r_[1., ma], seasonal_ma)[1:])


def difference_poly(order):
    _, d, _, _, seasonal_d, _ = order
    return np.convolve([1., -1.] if d else [1.],
                       np.r_[1., np.zeros(6), -1.] if seasonal_d else [1.])


class Arima:
    """Fit a regression with ARIMA errors by Gaussian conditional likelihood."""

    def __init__(self, order, weather=False):
        self.order, self.weather = order, weather
        self.delta = difference_poly(order)

    def fit(self, y, observed, temperature=None):
        degree = len(self.delta) - 1
        z = np.convolve(y, self.delta)[degree:len(y)]
        x = np.convolve(temperature, self.delta)[degree:len(y)] if self.weather else np.zeros(len(z))
        self.scale = z.std()
        z = z / self.scale
        # Require all days in the differencing window. Include the same
        # conditional burn-in in every candidate within each differencing group.
        clean = np.convolve(observed.astype(int), np.ones(degree+1))[degree:len(y)] == degree+1
        clean[:28] = False
        k = sum(self.order[i] for i in (0, 2, 3, 5))
        estimate_mean = degree == 0

        def components(raw):
            ar, ma = polynomials(raw, self.order)
            intercept = raw[k] if estimate_mean else 0.
            beta = raw[k + int(estimate_mean)] if self.weather else 0.
            return ar, ma, intercept, beta

        def objective(raw):
            ar, ma, intercept, beta = components(raw)
            errors = innovations(z - intercept - beta * x, ar, ma)[clean]
            return float(errors @ errors / len(errors))

        start = np.zeros(k + int(estimate_mean) + int(self.weather))
        if estimate_mean:
            start[k] = float(z[clean].mean())
        if self.weather:
            start[-1] = float(np.cov(z[clean], x[clean])[0, 1] / x[clean].var())
        raw, variance, self.converged = minimize(objective, start)
        self.ar, self.ma, intercept, beta = components(raw)
        self.intercept, self.beta = intercept * self.scale, beta * self.scale
        n, n_params = int(clean.sum()), len(raw) + 1
        variance *= self.scale**2
        deviance = n * (log(2 * np.pi * variance) + 1)
        self.aic = deviance + 2 * n_params
        self.bic = deviance + log(n) * n_params
        self.n = n
        return self

    def forecast(self, history, steps, temperature=None, future_temperature=None):
        degree = len(self.delta) - 1
        z = np.convolve(history, self.delta)[degree:len(history)]
        if self.weather:
            extended = np.r_[temperature, future_temperature]
            x = np.convolve(extended, self.delta)[degree:len(extended)]
        else:
            x = np.zeros(len(z) + steps)
        w = z - self.intercept - self.beta * x[:len(z)]
        errors = innovations(w, self.ar, self.ma).tolist()
        past = history.tolist()
        w = w.tolist()
        for h in range(steps):
            ar_part = sum(coef * w[-j] for j, coef in enumerate(self.ar, 1))
            ma_part = sum(coef * errors[-j] for j, coef in enumerate(self.ma, 1))
            value = ar_part + ma_part
            current = value + self.intercept + self.beta * x[len(z) + h]
            past.append(current - sum(c * past[-j] for j, c in enumerate(self.delta[1:], 1)))
            w.append(value)
            errors.append(0.)
        return np.array(past[-steps:])

    def residuals(self, history, observed, temperature):
        degree = len(self.delta)-1
        z = np.convolve(history, self.delta)[degree:len(history)]
        x = np.convolve(temperature, self.delta)[degree:len(history)] if self.weather else 0.
        e = innovations(z-self.intercept-self.beta*x, self.ar, self.ma)
        clean = np.convolve(observed.astype(int), np.ones(degree+1))[degree:len(history)] == degree+1
        e[~clean] = np.nan
        e[:28] = np.nan
        return np.r_[np.full(degree, np.nan), e]


class SeasonalNaive:
    def forecast(self, history, steps, *args):
        return np.resize(history[-7:], steps)

    def residuals(self, history, observed, _):
        e = history[7:] - history[:-7]
        e[~(observed[7:] & observed[:-7])] = np.nan
        return np.r_[np.full(7, np.nan), e]


def _hw_filter(y, p, state):
    y = np.ascontiguousarray(y, dtype=float)
    final_state = state.copy()
    errors = np.zeros(len(y))
    if _KERNELS is not None:
        _KERNELS.hw_sse(y, len(y), p, final_state, errors)
        return final_state, errors
    alpha, beta, gamma, phi = p
    level, slope = final_state[:2]
    for t, value in enumerate(y):
        idx = 2 + t % 7
        previous = level
        base = previous + phi * slope
        errors[t] = value - base - final_state[idx]
        level = alpha * (value - final_state[idx]) + (1-alpha) * base
        slope = beta * (level - previous) + (1-beta) * phi * slope
        final_state[idx] = gamma * (value - base) + (1-gamma) * final_state[idx]
    final_state[:2] = level, slope
    return final_state, errors


class HoltWinters:
    def __init__(self, damped):
        self.damped = damped

    def fit(self, y, observed):
        initial = y[:28].reshape(-1, 7)
        trend = (initial[-1].mean() - initial[0].mean()) / 21
        season = (initial - initial.mean(axis=1, keepdims=True)).mean(axis=0)
        self.initial = np.r_[initial[0].mean() - 4*trend, trend, season]
        scale = y.std()
        mask = observed.copy()
        mask[:28] = False

        def smooth(raw):
            a, b, g = 0.001 + .998 / (1 + np.exp(-np.clip(raw[:3], -20, 20)))
            phi = (.80 + .199 / (1 + np.exp(-np.clip(raw[3], -20, 20)))) if self.damped else 1.
            return np.array([a, b, g, phi])

        def objective(raw):
            _, errors = _hw_filter(y, smooth(raw), self.initial)
            return float(np.mean((errors[mask] / scale)**2))

        params, _, self.converged = minimize(objective, np.array([0., -2., -2., 1.])[:3+int(self.damped)])
        self.params = smooth(params)
        return self

    def forecast(self, history, steps, *args):
        state, _ = _hw_filter(history, self.params, self.initial)
        level, slope = state[:2]
        phi = self.params[-1]
        positions = (len(history) + np.arange(steps)) % 7
        return level + slope * np.cumsum(phi ** np.arange(1, steps+1)) + state[2+positions]

    def residuals(self, history, observed, _):
        _, errors = _hw_filter(history, self.params, self.initial)
        errors[~observed] = np.nan
        errors[:28] = np.nan
        return errors


def ols(y, x):
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    err = y - x @ coef
    return coef, err, float(np.sum(err**2))


def calendar_terms(day, harmonics=2):
    day = np.asarray(day)
    terms = [np.ones(len(day))]
    for h in range(1, harmonics+1):
        angle = 2 * np.pi * h * day / 365.2425
        terms.extend([np.sin(angle), np.cos(angle)])
    return np.column_stack(terms)


class TempForecast:
    """Two annual Fourier harmonics plus an autoregression on deviations."""

    def __init__(self, lag):
        self.lag = lag

    def fit(self, calendar_day, temperature, observed):
        features = calendar_terms(calendar_day)
        self.coef = ols(temperature[observed], features[observed])[0]
        anomaly = temperature - features @ self.coef
        i = np.arange(self.lag, len(temperature))
        i = i[np.logical_and.reduce([observed[i-j] for j in range(self.lag+1)])]
        columns = np.column_stack([anomaly[i-j] for j in range(1, self.lag+1)])
        self.ar = ols(anomaly[i], columns)[0]
        return self

    def forecast(self, day, temperature, steps):
        future = day[-1] + np.arange(1, steps+1)
        prev = (temperature[-self.lag:] - calendar_terms(day[-self.lag:]) @ self.coef).tolist()
        for _ in range(steps):
            prev.append(float(self.ar @ prev[-self.lag:][::-1]))
        return calendar_terms(future) @ self.coef + np.array(prev[-steps:])


def _beta_fraction(a, b, x):
    qab, qap, qam = a+b, a+1, a-1
    floor = 1e-280
    c, d = 1., 1-qab*x/qap
    d = 1 / (d if abs(d) > floor else floor)
    h = d
    for m in range(1, 501):
        for j in (0, 1):
            aa = (m*(b-m)*x/((qam+2*m)*(a+2*m)) if j == 0
                  else -(a+m)*(qab+m)*x/((a+2*m)*(qap+2*m)))
            d = 1 + aa*d
            c = 1 + aa/c
            d = 1 / (d if abs(d) > floor else floor)
            c = c if abs(c) > floor else floor
            change = d*c
            h *= change
        if abs(change-1) < 3e-13:
            break
    return h


def f_survival(value, df1, df2):
    if value <= 0:
        return 1.
    a, b = df2/2, df1/2
    x = df2 / (df2 + df1*value)
    common = exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*log(x)+b*log1p(-x))
    if x < (a+1)/(a+b+2):
        return common * _beta_fraction(a, b, x) / a
    return 1 - common * _beta_fraction(b, a, 1-x) / b


def chi_square_survival(value, df):
    x = value/2
    if df % 2:
        result, term, shape = erfc(sqrt(x)), 2*sqrt(x/np.pi)*exp(-x), .5
    else:
        result, term, shape = exp(-x), x*exp(-x), 1.
    while shape < df/2 - 1e-12:
        result += term
        shape += 1
        term *= x/shape
    return min(1., result)


def acf(y, max_lag):
    centered = y - np.mean(y)
    denominator = centered @ centered
    return np.array([1.] + [float(centered[k:] @ centered[:-k] / denominator)
                            for k in range(1, max_lag+1)])


def pacf(y, max_lag):
    r = acf(y, max_lag)
    return np.array([1.] + [np.linalg.solve(r[abs(np.arange(k)[:, None]-np.arange(k))],
                                          r[1:k+1])[-1] for k in range(1, max_lag+1)])


def adf(series, max_lag=14):
    """ADF with intercept; BIC lag selection on a common complete-row sample."""
    d = np.diff(series)
    i = np.arange(max_lag, len(d))
    i = i[np.isfinite(series[i-max_lag]) &
          np.logical_and.reduce([np.isfinite(d[i-j]) for j in range(max_lag+1)])]
    y = d[i]
    base = np.column_stack([np.ones(len(i)), series[i]])
    extra = np.column_stack([d[i-j] for j in range(1, max_lag+1)])
    candidates = []
    for lag in range(max_lag+1):
        x = np.column_stack([base, extra[:, :lag]])
        coef, residual, ssr = ols(y, x)
        n, k = x.shape
        candidates.append((n*log(ssr/n)+k*log(n), lag, coef, ssr, x))
    _, lag, coef, ssr, x = min(candidates, key=lambda a: a[0])
    n, k = x.shape
    se = sqrt(ssr/(n-k)*np.linalg.inv(x.T@x)[1, 1])
    statistic = float(coef[1]/se)
    critical = -2.86154 - 2.8903/n - 4.234/n**2 - 40.04/n**3
    return dict(statistic=statistic, lag=lag, n=n, critical_5=critical, reject=bool(statistic<critical))


def granger(target, temperature, day, weekday, max_lag=14):
    """Bivariate VAR lag choice, both directions, classical and HAC Wald tests."""
    dy, dx = np.diff(target), np.diff(temperature)
    i = np.arange(max_lag, len(dy))
    i = i[np.logical_and.reduce([np.isfinite(dy[i-j]) & np.isfinite(dx[i-j])
                                for j in range(max_lag+1)])]
    y = np.column_stack([dy[i], dx[i]])
    controls = np.column_stack([calendar_terms(day[i+1]), np.eye(7)[weekday[i+1], 1:]])
    this = np.column_stack([dy[i-j] for j in range(1, max_lag+1)])
    other = np.column_stack([dx[i-j] for j in range(1, max_lag+1)])
    fits = []
    for lag in range(1, max_lag+1):
        x = np.column_stack([controls, this[:, :lag], other[:, :lag]])
        coefs, e, _ = ols(y, x)
        n, k = x.shape
        sign, determinant = np.linalg.slogdet(e.T @ e / n)
        fits.append((determinant+2*k*log(n)/n, lag, x, coefs, e))
    _, lag, x, coef, e = min(fits, key=lambda a: a[0])
    n, k = x.shape
    inv = np.linalg.inv(x.T @ x)
    results = []
    for eq, direction in enumerate(("temperature_to_demand", "demand_to_temperature")):
        start = controls.shape[1] + (1-eq)*lag
        forbidden = np.arange(start, start+lag)
        keep = np.delete(np.arange(k), forbidden)
        ssr_full = float(e[:, eq] @ e[:, eq])
        ssr_restricted = ols(y[:, eq], x[:, keep])[2]
        f = max(0., (ssr_restricted-ssr_full)/lag/(ssr_full/(n-k)))
        scores = x * e[:, eq, None]
        score_at_day = np.zeros((len(dy), k))
        score_at_day[i] = scores
        meat = scores.T @ scores
        for delay in range(1, 15):
            product = score_at_day[delay:].T @ score_at_day[:-delay]
            meat += (1-delay/15)*(product+product.T)
        covariance = inv @ meat @ inv * n/(n-k)
        restriction = coef[forbidden, eq]
        wald = float(restriction @ np.linalg.solve(covariance[np.ix_(forbidden, forbidden)], restriction))
        results.append(dict(direction=direction, f=f, df1=lag, df2=n-k,
                            p=f_survival(f, lag, n-k), wald=wald,
                            p_hac=chi_square_survival(wald, lag), n=n))
    return results
