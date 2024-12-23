#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define N 8 // Размер сетки (должен быть степенью двойки)
#define n_iter 100 // Количество итераций
#define TOL 1e-6 // Порог для нормы разности

void initialize(double *f, int local_nx, int local_ny, int local_nz) {
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                f[i * local_ny * local_nz + j * local_nz + k] = rand() / (double)RAND_MAX;
            }
        }
    }
}

double compute_norm(double *f1, double *f2, int local_nx, int local_ny, int local_nz) {
    double norm = 0.0;
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                double diff = f1[i * local_ny * local_nz + j * local_nz + k] - f2[i * local_ny * local_nz + j * local_nz + k];
                norm += diff * diff;
            }
        }
    }
    return sqrt(norm);
}

int main(int argc, char **argv) {
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

    MPI_Datatype xy_plane, xz_plane, yz_plane;
    MPI_Type_vector(local_nx, local_ny, local_ny * local_nz, MPI_DOUBLE, &xy_plane);
    MPI_Type_commit(&xy_plane);
    MPI_Type_vector(local_nx, local_nz, local_ny * local_nz, MPI_DOUBLE, &xz_plane);
    MPI_Type_commit(&xz_plane);
    MPI_Type_vector(local_ny, local_nz, local_nz, MPI_DOUBLE, &yz_plane);
    MPI_Type_commit(&yz_plane);

    for (int iter = 0; iter < n_iter; iter++) {
        MPI_Request requests[12];
        MPI_Status statuses[12];

        // Обмен данными для плоскости XY
        MPI_Sendrecv(&f[local_ny * local_nz], 1, xy_plane, (rank + dims[0]) % dims[0], 0,
                     &f[0], 1, xy_plane, (rank - dims[0] + size) % dims[0], 0, cart_comm, &statuses[0]);
        MPI_Sendrecv(&f[(local_nx - 1) * local_ny * local_nz], 1, xy_plane, (rank - dims[0] + size) % dims[0], 0,
                     &f[local_nx * local_ny * local_nz], 1, xy_plane, (rank + dims[0]) % dims[0], 0, cart_comm, &statuses[1]);

        // Обмен данными для плоскости XZ
        MPI_Sendrecv(&f[local_ny * local_nz], 1, xz_plane, (rank + dims[1]) % dims[1], 0,
                     &f[0], 1, xz_plane, (rank - dims[1] + size) % dims[1], 0, cart_comm, &statuses[2]);
        MPI_Sendrecv(&f[(local_ny - 1) * local_nz], 1, xz_plane, (rank - dims[1] + size) % dims[1], 0,
                     &f[local_ny * local_nz], 1, xz_plane, (rank + dims[1]) % dims[1], 0, cart_comm, &statuses[3]);

        // Обмен данными для плоскости YZ
        MPI_Sendrecv(&f[local_nz], 1, yz_plane, (rank + dims[2]) % dims[2], 0,
                     &f[0], 1, yz_plane, (rank - dims[2] + size) % dims[2], 0, cart_comm, &statuses[4]);
        MPI_Sendrecv(&f[(local_nz - 1)], 1, yz_plane, (rank - dims[2] + size) % dims[2], 0,
                     &f[local_nz], 1, yz_plane, (rank + dims[2]) % dims[2], 0, cart_comm, &statuses[5]);


        // Метод Якоби
        for (int i = 0; i < local_nx; i++) {
            for (int j = 0; j < local_ny; j++) {
                for (int k = 0; k < local_nz; k++) {
                    double f1, f2, f3, f4, f5, f6;

                    // Верхняя и нижняя границы
                    if (i - 1 < 0) 
                        f1 = low_edge[j * local_nz + k];
                    else 
                        f1 = f[(i - 1) * local_ny * local_nz + j * local_nz + k];

                    if (i + 1 >= local_nx) 
                        f2 = up_edge[j * local_nz + k];
                    else 
                        f2 = f[(i + 1) * local_ny * local_nz + j * local_nz + k];

                    // Передняя и задняя границы
                    if (j - 1 < 0) 
                        f3 = back_edge[i * local_nz + k];
                    else 
                        f3 = f[i * local_ny * local_nz + (j - 1) * local_nz + k];

                    if (j + 1 >= local_ny) 
                        f4 = front_edge[i * local_nz + k];
                    else 
                        f4 = f[i * local_ny * local_nz + (j + 1) * local_nz + k];

                    // Левая и правая границы
                    if (k - 1 < 0) 
                        f5 = left_edge[i * local_ny + j];
                    else 
                        f5 = f[i * local_ny * local_nz + j * local_nz + (k - 1)];

                    if (k + 1 >= local_nz) 
                        f6 = right_edge[i * local_ny + j];
                    else 
                        f6 = f[i * local_ny * local_nz + j * local_nz + (k + 1)];

                    // Вычисление нового значения
                    f_new[i * local_ny * local_nz + j * local_nz + k] = (1.0/6.0) * (f1 + f2 + f3 + f4 + f5 + f6);
                }
            }
        }

        double *temp = f;
        f = f_new;
        f_new = temp;

        if (iter == n_iter - 1) {
            double local_norm = compute_norm(f, f_new, local_nx, local_ny, local_nz);
            double global_norm;
            MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, cart_comm);

            if (rank == 0) {
                printf("Норма разности между соседними решениями: %f\n", global_norm);
            }
        }
    }

    double end_time = MPI_Wtime();

    MPI_Type_free(&xy_plane);
    MPI_Type_free(&xz_plane);
    MPI_Type_free(&yz_plane);
    free(low_edge);
    free(up_edge);
    free(left_edge);
    free(right_edge);
    free(front_edge);
    free(back_edge);
    free(f);
    free(f_new);
    MPI_Finalize();
    return 0;
}
