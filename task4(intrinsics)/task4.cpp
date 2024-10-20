#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> 
#include <chrono>


void matrix_multiply_sequential(double *A, double *B, double *C, int num) 
{
    for (int i = 0; i < num; i++) 
    {
        for (int k = 0; k < num; k++) 
        {
            for (int j = 0; j < num; j++) 
            {
                C[i * num + j] += A[k * num + j] * B[j * num + k];
            }
        }
    }
}

void matrix_multiply_vectorized(double *A, double *B, double *C, int num) 
{
    for (int i = 0; i < num; i++) 
    {
        for (int j = 0; j < num; j += 4) 
        {
            __m256d c_vec = _mm256_setzero_pd(); 

            for (int k = 0; k < num; k++) 
            {
                __m256d a_vec = _mm256_load_pd(&A[k * num + j]); 
                __m256d b_vec = _mm256_broadcast_sd(&B[i * num + k]);

                c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec)); 
            }
            _mm256_store_pd(&C[i * num + j], c_vec); 
        }
    }
}


int main() {

    for (int num = 512; num <= 2048; num*=2)
    {
        printf("--------Size is %d--------\n", num);
        double* A = (double*) _mm_malloc(num * num * sizeof(double), 32);
        double* B = (double*) _mm_malloc(num * num * sizeof(double), 32);
        double *C_seq = (double*) _mm_malloc(num * num * sizeof(double), 32); 
        double *C_vec = (double*) _mm_malloc(num * num * sizeof(double), 32); 

        for (int i = 0; i < num * num; i++) 
        {
            A[i] = 0.01;
            B[i] = 0.01;
            C_seq[i] = 0.0;
            C_vec[i] = 0.0;
        }

        auto start1 = std::chrono::steady_clock::now();
        matrix_multiply_sequential(A, B, C_seq, num);
        auto end1 = std::chrono::steady_clock::now();
        auto elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
        printf("Sequential multiplication time: %lf sec\n", elapsed1.count()/ 1.0E6);

        auto start2 = std::chrono::steady_clock::now();
        matrix_multiply_vectorized(A, B, C_vec, num);
        auto end2 = std::chrono::steady_clock::now();
        auto elapsed2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
        printf("Vectorized multiplication time: %lf sec\n", elapsed2.count()/ 1.0E6);

        _mm_free(A);
        _mm_free(B);
        _mm_free(C_seq);
        _mm_free(C_vec);
    }

    return 0;
}

