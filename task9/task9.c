#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

void initialize_matrix(float* matrix, int N) 
{
    for (int i = 0; i < N * N; ++i) 
    {
        matrix[i] = (float)rand() / RAND_MAX; 
    }
}

void matrix_multiply_sequential(float *A, float *B, float *C, int num) 
{
    for (int i = 0; i < num; ++i) 
    {
        C[i] = 0.0;   
    }
    
    for (int i = 0; i < num; i++) 
    {
        for (int j = 0; j < num; j++) 
        {
            for (int k = 0; k < num; k++) 
            {
                C[i * num + j] += A[i * num + k] * B[k * num + j];
            }
         }
    }
}

int compare (float *C_seq, float *C, int num)
{
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < num; j++)
        {
            float dif = C_seq[i * num + j] - C[i * num + j];
            if (fabs(dif) < 1e-5) return 0;
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 576;
    int b = 2; 
    int sqrtP = (int)sqrt(size);

    if (N % sqrtP != 0 || (N / sqrtP) % b != 0) {
        if (rank == 0) {
            fprintf(stderr, "N must be divisible by sqrt(P) and (N/sqrt(P)) must be divisible by b.\n");
        }
        MPI_Finalize();
        return -1;
    }

    float* A = (float*)malloc((N / sqrtP) * N * sizeof(float));
    float* B = (float*)malloc(N * (N / sqrtP) * sizeof(float));
    float* C = (float*)malloc((N / sqrtP) * (N / sqrtP) * sizeof(float));
    float* A_col = (float*)malloc((N / sqrtP) * b * sizeof(float));
    float* B_row = (float*)malloc(b * (N / sqrtP) * sizeof(float));
    float* C_seq = (float*)malloc(N * N * sizeof(float));
    
    for (int i = 0; i < (N / sqrtP) * (N / sqrtP); ++i) 
    {
        C[i] = 0.0;  
    }

    double start_time = MPI_Wtime();

    if (rank == 0) {
        float* full_A = (float*)malloc(N * N * sizeof(float));
        float* full_B = (float*)malloc(N * N * sizeof(float));

        initialize_matrix(full_A, N);
        initialize_matrix(full_B, N);

        matrix_multiply_sequential(full_A, full_B, C_seq, N);
       
        MPI_Scatter(full_A, (N / sqrtP) * N, MPI_FLOAT, A, (N / sqrtP) * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(full_B, N * (N / sqrtP), MPI_FLOAT, B, N * (N / sqrtP), MPI_FLOAT, 0, MPI_COMM_WORLD);

        free(full_A);
        free(full_B);   
    } 
    else 
    {
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, A, (N / sqrtP) * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, B, N * (N / sqrtP), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }


    int dims[2] = {sqrtP, sqrtP};
    int periods[2] = {0, 0}; 
    MPI_Comm comm_cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);

    for (int k = 0; k < sqrtP; ++k) 
    {
        int coords[2];
        MPI_Cart_coords(comm_cart, rank, 2, coords);

        if (coords[0] == k) 
        {
            for (int i = 0; i < N / sqrtP; ++i) 
            {
                for (int j = 0; j < b; ++j) 
                {
                    A_col[i * b + j] = A[i * N + (k * b + j)];
                }
            }
        }
        MPI_Bcast(A_col, (N / sqrtP) * b, MPI_FLOAT, coords[0], comm_cart);

        if (coords[1] == k) 
        {
            for (int j = 0; j < N / sqrtP; ++j) 
            {
                for (int l = 0; l < b; ++l) 
                {
                    B_row[l * (N / sqrtP) + j] = B[(k * b + l) * (N / sqrtP) + j];
                }
            }
        }

        MPI_Bcast(B_row, b * (N / sqrtP), MPI_FLOAT, coords[1], comm_cart);

        for (int i = 0; i < N / sqrtP; ++i) 
        {
            for (int j = 0; j < N / sqrtP; ++j) 
            {
                for (int l = 0; l < b; ++l) 
                {
                    C[i * (N / sqrtP) + j] += A_col[i * b + l] * B_row[l * (N / sqrtP) + j];
                }
            }
        }
    }

    float* full_C = NULL;
    if (rank == 0) {
        full_C = (float*)malloc(N * N * sizeof(float));
    }

    MPI_Gather(C, (N / sqrtP) * (N / sqrtP), MPI_FLOAT, full_C, (N / sqrtP) * (N / sqrtP), MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        
        if (compare(C_seq, full_C, N)) 
            printf("Matrixes are equal\n");
        else 
            printf("Matrixes are not equal\n");
        free(full_C);
    }

    free(A);
    free(B);
    free(C);
    free(A_col);
    free(B_row);
    
    double end_time = MPI_Wtime() - start_time;
    double max_time;
    MPI_Reduce(&end_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {
        printf("Time of execution: %f sec\n", max_time);
    }

    MPI_Finalize();
    return 0;
}