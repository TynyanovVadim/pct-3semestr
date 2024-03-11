#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <time.h>

void run_parallel(int m, int n);
void run_serial(int m, int n);

int main() 
{
    int n, m;
    n = m = 0;

    printf ("enter n m: ");
    scanf("%d %d", &n, &m);
    printf("\n");

    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);

    printf ("serial: \n");
    run_serial(m, n);
    
    for (int i = 2; i <= omp_get_num_threads(); i += 2) {
        printf ("Threads: %d --- ", i);
        omp_set_num_threads(i);
        run_parallel(m, n);
    }

    return 0;
}

double wtime()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1E-9;
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;

        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_parallel(int m, int n)
{
    double *a, *b, *c;
    // Allocate memory for 2-d array a[m, n]
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = malloc(sizeof(*c) * m);

    assert(a && "Memory error(a), parallel");
    assert(b && "Memory error(b), parallel");
    assert(c && "Memory error(c), parallel");

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++)
                a[i * n + j] = i + j;
            c[i] = 0.0;
        
        }
    }

    for (int j = 0; j < n; j++)
        b[j] = j;   
     
    double t = wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    t = wtime() - t;

    printf("Elapsed time (parallel): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n)
{
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void run_serial(int m, int n)
{
    double *a, *b, *c;
    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = malloc(sizeof(*c) * m);

    assert(a && "Memory error(a), serial");
    assert(b && "Memory error(b), serial");
    assert(c && "Memory error(c), serial");

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
        }

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = wtime();
    matrix_vector_product(a, b, c, m, n);
    t = wtime() - t;

    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}