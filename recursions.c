/* Optional acceleration for the two scalar recursions in models.py. */
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void innovations(const double *z, int n, const double *ar, int p,
                        const double *ma, int q, double *err) {
    for (int t = 0; t < n; ++t) {
        double value = z[t];
        for (int j = 0; j < p && j < t; ++j) value -= ar[j] * z[t-j-1];
        for (int j = 0; j < q && j < t; ++j) value -= ma[j] * err[t-j-1];
        err[t] = value;
    }
}

EXPORT void hw_sse(const double *y, int n, const double *p,
                   double *state, double *errors) {
    double level = state[0], slope = state[1];
    for (int t = 0; t < n; ++t) {
        int j = 2 + t % 7;
        double base = level + p[3] * slope;
        double next_level = p[0] * (y[t] - state[j]) + (1-p[0]) * base;
        errors[t] = y[t] - base - state[j];
        slope = p[1] * (next_level-level) + (1-p[1]) * p[3] * slope;
        state[j] = p[2] * (y[t]-base) + (1-p[2]) * state[j];
        level = next_level;
    }
    state[0] = level;
    state[1] = slope;
}
