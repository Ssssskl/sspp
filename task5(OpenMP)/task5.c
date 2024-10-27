#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main() {
    int a, b, x, N;
    double p;

    printf("Enter a, b, x, p, N: ");
    scanf("%d %d %d %lf %d", &a, &b, &x, &p, &N);

    int count_b = 0;
    double total_time = 0.0;

    double start;
    double end;

    start = omp_get_wtime();

    #pragma omp parallel
    {

        unsigned int seed = omp_get_thread_num();

        #pragma omp for 
            for (int i = 0; i < N; i++) 
            {
                int position = x;

                double start1, end1;
                start1 = omp_get_wtime();

                while (position > a && position < b) 
                {
                    double rand_val = (double)rand_r(&seed) / RAND_MAX;
                    if (rand_val < p) {
                        position++; 
                    } 
                    else {
                        position--; 
                    }
                }

                end1 = omp_get_wtime();

                if (position == b)
                {
                    #pragma omp critical
                    {
                        count_b++;
                    }
                        
                }

                #pragma omp critical
                {
                    total_time += (end1 - start1); 
                }     
            }
    }
    end = omp_get_wtime();

    double probability_b = (double) count_b / N;
    double average_time = total_time / N;

    printf("Probability of reaching b: %f\n", probability_b);
    printf("Average lifetime of a particle: %f seconds\n", average_time);
    printf("Time taken for the main loop: %f seconds\n", end - start);

    return 0;
}