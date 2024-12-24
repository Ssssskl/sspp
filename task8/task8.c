#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int N = 1024;
int K = 1000;

void initialize_grid(int *arr_old, int N, int num_str, int rank) 
{
    for (int i = 0; i < num_str; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            arr_old[i * N + j] = rand() % 2; 
        }
    }
}

void exchange_boundaries(int *arr_old, int *arr_up, int *arr_down, int N, int num_str, int rank, int size) 
{
    MPI_Request reqs[4];
    int cnt = 0;

    if (size != 1) 
    {
        if (rank > 0) 
        {
            MPI_Irecv(arr_up, N, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
            MPI_Isend(arr_old, N, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
        }
        if (rank < size - 1) 
        {
            MPI_Irecv(arr_down, N, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
            MPI_Isend(arr_old + N * (num_str - 1), N, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &reqs[cnt++]);
        }
    }

    MPI_Waitall(cnt, reqs, MPI_STATUSES_IGNORE);
}

void update_grid(int *arr_old, int *arr_new, int N, int num_str, int *num_in_part_new, int arr_up[], int arr_down[], int rank, int size) 
{
    *num_in_part_new = 0;

    for (int i = 0; i < num_str; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            int neighbors = 0;

            for (int x = -1; x <= 1; x++) 
            {
                for (int y = -1; y <= 1; y++) 
                {
                    if (x == 0 && y == 0) 
                        continue; 

                    if ( (i + x >= 0 && i + x< num_str) && (j + y >= 0 && j + y < N)) 
                    {
                        neighbors += arr_old[(i + x) * N + (j + y)];
                    }
                    if (i + x < 0 && rank != 0 && (j + y >= 0 && j + y < N)) 
                    {
                        neighbors += arr_up[j + y];
                    }
                    if (i + x == num_str && rank != (size - 1) && j + y >= 0 && j + y < N) 
                    {
                        neighbors += arr_down[j + y];
                    }
                }
            }

            if (arr_old[i * N + j]) 
            {
                arr_new[i * N + j] = (neighbors == 2 || neighbors == 3) ? 1 : 0; 
            } 
            else
            {
                arr_new[i * N + j] = (neighbors == 3) ? 1 : 0;
            } 

            if (arr_new[i * N + j] == 1) 
            {
                (*num_in_part_new)++;
            }
        }
    }
}

int main(int argc, char *argv[]) 
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double start_time = MPI_Wtime();
    
    int num_elem = (N * N) / size; 
    int num_str = N / size;

    int* arr_old = (int*)malloc(num_elem * sizeof(int));
    int* arr_new = (int*)malloc(num_elem * sizeof(int));

    initialize_grid(arr_old, N, num_str, rank);

    int* arr_up = (int*)malloc(N * sizeof(int));
    int* arr_down = (int*)malloc(N * sizeof(int));

    int flag;
    int num_in_part_new = 0, num_in_part_old = 0, num;

    update_grid(arr_old, arr_new, N, num_str, &num_in_part_new, arr_up, arr_down, rank, size);
    num_in_part_old = num_in_part_new;

    while (1) 
    {
        exchange_boundaries(arr_old, arr_up, arr_down, N, num_str, rank, size);
        update_grid(arr_old, arr_new, N, num_str, &num_in_part_new, arr_up, arr_down, rank, size);

        int is_stable = 0;

        if (K <= 0) 
        {
            if (num_in_part_old == num_in_part_new) 
                is_stable = 1;
        }

        MPI_Allreduce(&is_stable, &flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        
        if (flag == 1) 
            break;
        
        for (int i = 0; i < num_elem; i++)
            arr_old[i] = arr_new[i];
        
        num_in_part_old = num_in_part_new; 
        K--;
    }

    free(arr_new);
    free(arr_old);
    free(arr_up);
    free(arr_down);

    double end_time = MPI_Wtime() - start_time;
    double max_time;
    MPI_Reduce(&num_in_part_new, &num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) 
    {
        printf("Число живых клеток: %d\n", num);
        printf("Время выполнения: %f сек\n", max_time);
    }

    MPI_Finalize();
    return 0;
}