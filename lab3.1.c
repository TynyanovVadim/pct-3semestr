#include <math.h>
#include <omp.h>
#include <stdio.h>

double
func(double x)
{
    return x / pow(sin(2 * x), 3);
    // return 5;
}

int main()
{
    const double eps = 1E-5;
    const double a = 0.1;
    const double b = 0.5;
    const int n0 = 100000000;

    printf("Numerical integration: [%f, %f], n0 = %d, EPS = %f\n", a, b, n0, eps);

    FILE *f = fopen ("midpoint.csv", "w");

    int threads[] = {1, 2, 4, 6, 8};
    double t1 = 0;
    for (int i = 0; i < 5; i++) {
        int p = threads[i];
        double t = omp_get_wtime();
        
        double sq[2];
        #pragma omp parallel num_threads (p)
        {
            int n = n0, k;
            double delta = 1;
            for (k = 0; delta > eps; n *= 2, k ^= 1) {
                double h = (b - a) / n;
                double s = 0.0;
                sq[k] = 0;
                #pragma omp barrier

                #pragma omp for nowait
                for (int i = 0; i < n; i++)
                    s += func(a + h * (i + 0.5));

                #pragma omp atomic
                    sq[k] += s * h;
                #pragma omp barrier

                if (n > n0)
                    delta = fabs(sq[k] - sq[k ^ 1]) / 3.0;
            }

            #pragma omp master
                printf("Result : %.12f; Runge rule: EPS %e, n %d\n", sq[k], eps, n / 2);
        }

        t = omp_get_wtime() - t;
        if (p == 1) {
            t1 = t;
        }
        printf("Thereads: %d; Elapsed time (sec.): %.6f\n", p, t);
        printf("Speedup (threads: %d): %.6f\n", p, t1/t);

        fprintf(f, "%d; %.6f; %.6f\n", p, t, t1/t);

    }
    return 0;
}
