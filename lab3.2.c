#define _POSIX_C_SOURCE 1

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double getrand(unsigned int *seed)
{
    return (double)rand_r(seed) / RAND_MAX;
}

double
func(double x, double y)
{
    return pow(exp(x + y), 2);
    // return 5;
}

int main()
{
    const int n = 10000000;
    printf("Numerical integration by Monte Carlo method: n = %d\n", n);

    int in = 0;
    double s = 0;

    FILE *f = fopen ("MonteCarlo.csv", "w");

    int threads[] = {1, 2, 4, 6, 8};
    double t1 = 0;
    for (int i = 0; i < 5; i++) {
        int p = threads[i];
        double t = omp_get_wtime();

        in = 0, s = 0;
        #pragma omp parallel num_threads (p)
        {
            double s_loc = 0;
            int in_loc = 0;
            unsigned int seed = omp_get_thread_num();

            #pragma omp for nowait
            for (int i = 0; i < n; i++) {
                double x = getrand(&seed); /* x in (0, 1) */
                double y = getrand(&seed); /* y in (0, 1 - x) */
                if (x > 0 && x < 1 &&
                    y > 0 && y < (1 - x)
                    ) {
                    in_loc++;
                    s_loc += func(x, y);
                }
            }

            #pragma omp atomic
                s += s_loc;
            #pragma omp atomic
                in += in_loc;
        }

        t = omp_get_wtime() - t;
        if (p == 1) {
            t1 = t;
        }
        printf("Thereads: %d; Elapsed time (sec.): %.6f\n", p, t);
        printf("Speedup (threads: %d): %.6f\n", p, t1/t);

        fprintf(f, "%d; %.6f; %.6f\n", p, t, t1/t);

    }

    double v = 0.5;
    double res = v * s / in;
    printf("Result: %.12f, n %d\n", res, n);

    return 0;
}
