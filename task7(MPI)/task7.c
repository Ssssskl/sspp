#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


#define N_ITER 10000

void initialize_grid(double *grid, int local_rows, int N) 
{
    for (int i = 0; i < local_rows; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            grid[i * N + j] = rand() % 100; 
        }
    }
}

double compute_norm(double *old_grid, double *new_grid, int local_rows, int N) 
{
    double norm = 0.0;
    for (int i = 0; i < local_rows; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            double diff = new_grid[i * N + j] - old_grid[i * N + j];
            norm += diff * diff;

        }
    }
    return sqrt(norm); 
}

int main(int argc, char *argv[]) 
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = MPI_Wtime();

    int N = atoi(argv[1]);
    int num_elem = (N * N) / size;
    int local_rows = N / size;

    double *grid = (double *)malloc(local_rows * N * sizeof(double));
    double *new_grid = (double *)malloc(local_rows * N * sizeof(double));
    double *send_up = (double *)malloc(N * sizeof(double));
    double *send_down = (double *)malloc(N * sizeof(double));
    double norm = 0.0, norm_i = 0.0;
    
    initialize_grid(grid, local_rows, N);

    for (int iter = 0; iter < N_ITER; iter++) 
    {
        if (size != 1)
        {
            if (rank == 0) 
            {
                MPI_Sendrecv(grid + N * (local_rows - 1), N, MPI_DOUBLE, 1, 0,
                send_down, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else if (rank == size - 1) 
            {
                MPI_Sendrecv(grid, N, MPI_DOUBLE, rank - 1, 0,
                send_up, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                MPI_Sendrecv(grid, N, MPI_DOUBLE, rank - 1, 0, 
                send_up, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(grid + (local_rows - 1) * N, N, MPI_DOUBLE, rank + 1, 0, 
                send_down, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        for (int i = 0; i < local_rows; i++) 
        {
            for (int j = 0; j < N; j++) 
            {
                double up, down, left, right;
                if (i == 0)
                {
                    if (rank == 0)
                        up = 0.0;
                    else
                        up = send_up[j];   
                }
                else
                    up = grid[(i - 1) * N + j];
                if (i == local_rows - 1)
                {
                    if (rank == size-1)
                        down = 0.0;
                    else
                        down = send_down[j];
                } 
                else 
                    down = grid[N*(i+1) + j];

                left = (j == 0) ? 0.0 : grid[i * N + (j - 1)];
                right = (j == N - 1) ? 0.0 : grid[i * N + (j + 1)];

                new_grid[i * N + j] = 0.25 * (up + down + left + right);
            }
        }
        

        if (iter == N_ITER - 1) 
        {
            double norm = compute_norm(grid, new_grid, local_rows, N);
            double fin_norm;

            printf("Rank: %d, Final norm of difference: %f\n", rank, fin_norm);
        } 
        else
        {
            for (int i = 0; i < num_elem; i++)
                grid[i] = new_grid[i];
        }

    }


    free(grid);
    free(new_grid);
    free(send_up);
    free(send_down);

    double end = MPI_Wtime() - start;
    double time;
    MPI_Reduce(&end, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        printf("Time of the program: %f sec\n", time);
    }
    
    MPI_Finalize();
    return 0;
}