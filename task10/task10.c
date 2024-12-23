#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define N 64 
#define n_iter 20000 
#define TOL 1e-6 

void initialize(double *f, int local_nx, int local_ny, int local_nz) 
{
    for (int i = 0; i < local_nx; i++) 
    {
        for (int j = 0; j < local_ny; j++) 
        {
            for (int k = 0; k < local_nz; k++) 
            {
                f[i * local_ny * local_nz + j * local_nz + k] = rand() / (double)RAND_MAX;
            }
        }
    }
}

double compute_norm(double *f1, double *f2, int local_nx, int local_ny, int local_nz) 
{
    double norm = 0.0;
    for (int i = 0; i < local_nx; i++) 
    {
        for (int j = 0; j < local_ny; j++) 
        {
            for (int k = 0; k < local_nz; k++) 
            {
                double diff = f1[i * local_ny * local_nz + j * local_nz + k] - f2[i * local_ny * local_nz + j * local_nz + k];
                norm += diff * diff;
            }
        }
    }
    return sqrt(norm);
}

int main(int argc, char **argv) 
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    int dims[3] = {0, 0, 0};
    int periods[3] = {0, 0, 0};
    MPI_Dims_create(size, 3, dims);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int local_nx = N / dims[0];
    int local_ny = N / dims[1];
    int local_nz = N / dims[2];

    double *f = (double *)malloc(local_nx * local_ny * local_nz * sizeof(double));
    double *f_new = (double *)malloc(local_nx * local_ny * local_nz * sizeof(double));

    initialize(f, local_nx, local_ny, local_nz);

    double *low_edge = (double *)calloc(local_ny * local_nz, sizeof(double));
    double *up_edge = (double *)calloc(local_ny * local_nz, sizeof(double));
    double *left_edge = (double *)calloc(local_nx * local_nz, sizeof(double));
    double *right_edge = (double *)calloc(local_nx * local_nz, sizeof(double));
    double *front_edge = (double *)calloc(local_nx * local_ny, sizeof(double));
    double *back_edge = (double *)calloc(local_nx * local_ny, sizeof(double));

    int neighbors[6];
    MPI_Cart_shift(cart_comm, 0, 1, &neighbors[0], &neighbors[1]); // Low, Up
    MPI_Cart_shift(cart_comm, 1, 1, &neighbors[2], &neighbors[3]); // Left, Right
    MPI_Cart_shift(cart_comm, 2, 1, &neighbors[4], &neighbors[5]); // Front, Back

    for (int iter = 0; iter < n_iter; iter++) 
    {
        MPI_Request requests[12];

        MPI_Irecv(low_edge, local_ny * local_nz, MPI_DOUBLE, neighbors[0], 0, cart_comm, &requests[0]);
        MPI_Isend(&f[0], local_ny * local_nz, MPI_DOUBLE, neighbors[0], 0, cart_comm, &requests[1]);

        MPI_Irecv(up_edge, local_ny * local_nz, MPI_DOUBLE, neighbors[1], 0, cart_comm, &requests[2]);
        MPI_Isend(&f[(local_nx - 1) * local_ny * local_nz], local_ny * local_nz, MPI_DOUBLE, neighbors[1], 0, cart_comm, &requests[3]);

        MPI_Irecv(left_edge, local_nx * local_nz, MPI_DOUBLE, neighbors[2], 0, cart_comm, &requests[4]);
        MPI_Isend(&f[0], local_nx * local_nz, MPI_DOUBLE, neighbors[2], 0, cart_comm, &requests[5]);

        MPI_Irecv(right_edge, local_nx * local_nz, MPI_DOUBLE, neighbors[3], 0, cart_comm, &requests[6]);
        MPI_Isend(&f[(local_ny - 1) * local_nz], local_nx * local_nz, MPI_DOUBLE, neighbors[3], 0, cart_comm, &requests[7]);

        MPI_Irecv(front_edge, local_nx * local_ny, MPI_DOUBLE, neighbors[4], 0, cart_comm, &requests[8]);
        MPI_Isend(&f[0], local_nx * local_ny, MPI_DOUBLE, neighbors[4], 0, cart_comm, &requests[9]);

        MPI_Irecv(back_edge, local_nx * local_ny, MPI_DOUBLE, neighbors[5], 0, cart_comm, &requests[10]);
        MPI_Isend(&f[local_nz - 1], local_nx * local_ny, MPI_DOUBLE, neighbors[5], 0, cart_comm, &requests[11]);

        MPI_Waitall(12, requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < local_nx; i++) 
        {
            for (int j = 0; j < local_ny; j++) 
            {
                for (int k = 0; k < local_nz; k++) 
                {
                    double f1, f2, f3, f4, f5, f6;

                    f1 = (i == 0) ? low_edge[j * local_nz + k] : f[(i - 1) * local_ny * local_nz + j * local_nz + k];
                    f2 = (i == local_nx - 1) ? up_edge[j * local_nz + k] : f[(i + 1) * local_ny * local_nz + j * local_nz + k];

                    f3 = (j == 0) ? back_edge[i * local_nz + k] : f[i * local_ny * local_nz + (j - 1) * local_nz + k];
                    f4 = (j == local_ny - 1) ? front_edge[i * local_nz + k] : f[i * local_ny * local_nz + (j + 1) * local_nz + k];

                    f5 = (k == 0) ? left_edge[i * local_ny + j] : f[i * local_ny * local_nz + j * local_nz + (k - 1)];
                    f6 = (k == local_nz - 1) ? right_edge[i * local_ny + j] : f[i * local_ny * local_nz + j * local_nz + (k + 1)];

                    f_new[i * local_ny * local_nz + j * local_nz + k] = (1.0 / 6.0) * (f1 + f2 + f3 + f4 + f5 + f6);
                }
            }
        }

        if (iter == n_iter - 1) 
        {
            double local_norm = compute_norm(f, f_new, local_nx, local_ny, local_nz);

            double global_norm;
            MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

            if (rank == 0) 
            {
                printf("The norm of the difference between neighboring solutions: %f\n", global_norm);
            }
        }

        double *temp = f;
        f = f_new;
        f_new = temp;
    }

    double end_time = MPI_Wtime() - start_time;
    double max_time;
    MPI_Reduce(&end_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        printf("Time of execution: %f sec\n", max_time);
    }

    free(f);
    free(f_new);
    free(low_edge);
    free(up_edge);
    free(left_edge);
    free(right_edge);
    free(front_edge);
    free(back_edge);
    MPI_Finalize();
    return 0;
}
