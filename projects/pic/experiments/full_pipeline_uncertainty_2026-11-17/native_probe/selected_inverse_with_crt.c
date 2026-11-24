#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Exact same selected-inverse recurrence as score_core.py.
   Arrays use row-major storage [bandwidth + 1, n]. */
__declspec(dllexport) int selected_inverse(const double *chol, double *inv,
                                          size_t n, size_t width) {
    double *weights = (double *)malloc((width + 1) * sizeof(double));
    double *off = (double *)malloc((width + 1) * sizeof(double));
    if (!weights || !off) { free(weights); free(off); return 1; }
    memset(inv, 0, (width + 1) * n * sizeof(double));
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
    free(weights); free(off); return 0;
}
