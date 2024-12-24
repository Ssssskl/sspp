#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void initialize_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = i; 
    }
}

void multiply_blocks(float *A, float *B, float *C, int block_size, int b) {
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            for (int k = 0; k < b; k++) {
                C[i * block_size + j] += A[i * b + k] * B[k * block_size + j];
            }
        }
    }
}

int compare_results(float* arr1, float* arr2, int N) {
    for (int i = 0; i < N; i++)
        if (fabs(arr1[i] - arr2[i]) > 1e-5) 
            return 0;
    return 1;
}

void send_blocks(float* A, float* B, int N, int block_size, int sqrt_p, MPI_Comm cart_comm) {
    for (int i = 0; i < sqrt_p; i++) {
        for (int j = 0; j < sqrt_p; j++) {
            if (i == 0 && j == 0) continue; 
            const int coords[2] = {i, j};
            int rank_cart;
            MPI_Cart_rank(cart_comm, coords, &rank_cart);
            MPI_Send(A + i * N * block_size + j * block_size, 1, MPI_FLOAT, rank_cart, 0, cart_comm);
            MPI_Send(B + i * N * block_size + j * block_size, 1, MPI_FLOAT, rank_cart, 1, cart_comm);
        }
    }
}

void receive_blocks(float* A_block, float* B_block, int block_size, MPI_Comm cart_comm) {
    MPI_Recv(A_block, block_size * block_size, MPI_FLOAT, 0, 0, cart_comm, MPI_STATUS_IGNORE);
    MPI_Recv(B_block, block_size * block_size, MPI_FLOAT, 0, 1, cart_comm, MPI_STATUS_IGNORE);
}

void gather_results(float* C, float* C_block, int N, int block_size, int sqrt_p, MPI_Comm cart_comm) {
    for (int i = 0; i < sqrt_p; i++) 
    {
        for (int j = 0; j < sqrt_p; j++) 
        {
            if (i == 0 && j == 0) continue; 
            const int coords[2] = {i, j};
            int rank_cart;
            MPI_Cart_rank(cart_comm, coords, &rank_cart);
            MPI_Recv(C + i * N * block_size + j * block_size, block_size * block_size, MPI_FLOAT, rank_cart, 0, cart_comm, MPI_STATUS_IGNORE);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double start = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <N> <b>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);       
    int b = atoi(argv[2]);   
    int P = size;              
    int sqrt_p = (int)sqrt(P);    
    int block_size = N / sqrt_p;

    int dims[2] = {sqrt_p, sqrt_p};
    int periods[2] = {0, 0}; 
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    float* A_block = (float*)malloc(block_size * block_size * sizeof(float));
    float* B_block = (float*)malloc(block_size * block_size * sizeof(float));
    float* C_block = (float*)malloc(block_size * block_size * sizeof(float));

    MPI_Datatype block_type;
    MPI_Type_vector(block_size, block_size, N, MPI_FLOAT, &block_type);
    MPI_Type_commit(&block_type);

    float* A = NULL;
    float* B = NULL;
    float* C = NULL;

    if (rank == 0) 
    {
        A = (float*)malloc(N * N * sizeof(float));
        B = (float*)malloc(N * N * sizeof(float));
        initialize_matrix(A, N, N);
        initialize_matrix(B, N, N);

        for (int i = 0; i < block_size; i++) { 
            for (int j = 0; j < block_size; j++) {
                A_block[i * block_size + j] = A[i * N + j];
                B_block[i * block_size + j] = B[i * N + j];
            }
        }

        if (size != 1) 
        {
            send_blocks(A, B, N, block_size, sqrt_p, cart_comm);
        }
    } else 
    {
        receive_blocks(A_block, B_block, block_size, cart_comm);
    }

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int row_rank = coords[0]; 
    int col_rank = coords[1]; 

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart_comm, row_rank, 0, &row_comm);
    MPI_Comm_split(cart_comm, col_rank, 0, &col_comm);
    
    float* A_i_k = (float*)malloc(b * block_size * sizeof(float));
    float* B_k_j = (float*)malloc(b * block_size * sizeof(float));
    memset(C_block, 0, block_size * block_size * sizeof(float));

    for (int k = 0; k < N / b; k++) 
    {
        int A_root = (k * b) / block_size;       
        int B_root = (k * b) / block_size; 

        if (col_rank == A_root) 
        {
            for (int i = 0; i < block_size; i++) 
            {
                for (int j = 0; j < b; j++) {
                    A_i_k[i * b + j] = A_block[(k * b) % block_size + i * block_size + j];
                }
            }
        }
        if (size != 1) 
        {
            MPI_Bcast(A_i_k, block_size * b, MPI_FLOAT, A_root, row_comm);
        }
         
        if (row_rank == B_root) 
        {
            for (int i = 0; i < b; i++) 
            {
                for (int j = 0; j < block_size; j++) 
                {
                    B_k_j[i * block_size + j] = B_block[(k * b) % block_size * block_size + i * block_size + j];
                }
            }
        }
        if (size != 1)
            MPI_Bcast(B_k_j, b * block_size, MPI_FLOAT, B_root, col_comm);

        multiply_blocks(A_i_k, B_k_j, C_block, block_size, b);
    }
    
    if (rank == 0) 
    {
        C = (float*)malloc(N * N * sizeof(float));

        for (int i = 0; i < block_size; i++) 
        {
            for (int j = 0; j < block_size; j++) 
            {
                C[i * N + j] = C_block[i * block_size + j];
            }
        }

        if (size != 1) 
        {
            gather_results(C, C_block, N, block_size, sqrt_p, cart_comm);
        }
        
        double time = MPI_Wtime() - start;
        printf("Время выполнения: %f секунд\n", time);

        float* C_seq = (float*)malloc(N * N * sizeof(float));
        memset(C_seq, 0, N * N * sizeof(float));
        multiply_blocks(A, B, C_seq, N, N);
        printf("Результат совпадают?: %s\n", compare_results(C, C_seq, N*N) ? "Да" : "Нет");
        free(C_seq);
        free(A);
        free(B);
        free(C);
    } 
    else 
    {
        MPI_Send(C_block, block_size * block_size, MPI_FLOAT, 0, 0, cart_comm);
    }
    
    free(A_i_k);
    free(B_k_j);
    free(A_block);
    free(B_block);
    free(C_block);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Type_free(&block_type);
    MPI_Finalize();
    return 0;
}