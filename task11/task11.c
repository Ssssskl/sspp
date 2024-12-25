#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void initialize_matrix(double *A, int rows, int cols, int rank) 
{
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            A[i * cols + j] = (double)(rank * 2 + 1); 
        }
    }
}

void initialize_vector(double *b, int size) 
{
    for (int i = 0; i < size; i++)
    {
        b[i] = (double)(i + 1); 
    }
}

void print_vector(double *c, int size) 
{
    for (int i = 0; i < size; i++) 
    {
        printf("%f ", c[i]);
    }
    printf("\n");
}

int main(int argc, char **argv) 
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0}; 
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    int N = 24000; 
    int rows = N / dims[0]; 
    int cols = N / dims[1];

    double start_time = MPI_Wtime();

    double *local_A = (double *)malloc(rows * cols * sizeof(double));
    double *local_c = (double *)malloc(N * sizeof(double));
    double *b = NULL;

    for (int i = 0; i < N; i++) 
    {
        local_c[i] = 0.0;
    }

    if (rank == 0) 
    {
        b = (double *)malloc(N * sizeof(double));
        initialize_vector(b, N);
        //print_vector(b, N);
    }

    MPI_Win win_b;

    if (size > 1) 
    {
        if (rank == 0) 
        {
            MPI_Win_create(b, N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_b);
        } 
        else 
        {
            MPI_Win_create(NULL, 0, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_b);
        }

        MPI_Win_fence(0, win_b);
    }

    initialize_matrix(local_A, rows, cols, rank);
    // printf("Matrix\n");
    // print_vector(local_A, rows * N);

    double *b_local = (double *)malloc(cols * sizeof(double));

    if (size > 1)
    {
        MPI_Get(b_local, cols, MPI_DOUBLE, 0, coords[1] * cols, cols, MPI_DOUBLE, win_b);
        MPI_Win_fence(0, win_b);
    } 
    else 
    {
        for (int i = 0; i < cols; i++) 
        {
            b_local[i] = b[i];
        }
    }

    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            local_c[coords[0]*rows + i] += local_A[i * cols + j] * b_local[j];
        }
    }

    double *c = NULL;
    if (rank == 0) 
    {
        c = (double *)malloc(N * sizeof(double));
    }

    MPI_Reduce(local_c, c, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        //print_vector(c, N);
        free(c);
    }

    double end_time = MPI_Wtime();
    double max_time;
    MPI_Reduce(&end_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        printf("Time of execution: %f sec\n", max_time);
    }

    free(local_A);
    free(local_c);
    free(b_local);
    if (rank == 0) 
    {
        free(b);
    }
    if (size > 1) 
    {
        MPI_Win_free(&win_b);
    }

    MPI_Finalize();
    return 0;
}