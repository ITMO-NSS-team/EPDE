/* Windows x64 kernel without a C runtime dependency. The Python caller owns
   and zero-initializes all storage; no allocation or operating-system calls. */
typedef unsigned long long size_t;
int _fltused = 0;

/* Exact same selected-inverse recurrence as score_core.py.
   Arrays use row-major storage [bandwidth + 1, n]. */
__declspec(dllexport) int selected_inverse(const double *chol, double *inv,
                                          size_t n, size_t width,
                                          double *weights, double *off) {
    for (size_t ii = n; ii > 0; --ii) {
        size_t i = ii - 1;
        size_t k = width < n - 1 - i ? width : n - 1 - i;
        for (size_t j = 1; j <= k; ++j)
            weights[j] = chol[j * n + i] / chol[i];
        for (size_t j = 1; j <= k; ++j) {
            double value = 0.;
            for (size_t ell = 1; ell <= k; ++ell) {
                size_t lag = j > ell ? j - ell : ell - j;
                size_t col = i + (j < ell ? j : ell);
                value += inv[lag * n + col] * weights[ell];
            }
            off[j] = -value;
        }
        double value = 0.;
        for (size_t j = 1; j <= k; ++j) {
            inv[j * n + i] = off[j];
            value += weights[j] * off[j];
        }
        inv[i] = 1. / (chol[i] * chol[i]) - value;
    }
    return 0;
}
