#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void merge(int* array, int left, int mid, int right) 
{
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = (int*)malloc(n1 * sizeof(int));
    int* R = (int*)malloc(n2 * sizeof(int));

    for (i = left; i < mid + 1; i++)
        L[i - left] = array[i];
    for (j = mid + 1; j < right + 1; j++)
        R[j - mid - 1] = array[j];

    i = 0;
    j = 0; 
    k = left; 
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            array[k] = L[i];
            i++;
        } else {
            array[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        array[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        array[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

void heapify(int* array, int n, int i) 
{
    int largest = i; 
    int left = 2 * i + 1; 
    int right = 2 * i + 2; 

    if (left < n && array[left] > array[largest])
        largest = left;

    if (right < n && array[right] > array[largest])
        largest = right;

    if (largest != i) 
    {
        int temp = array[i];
        array[i] = array[largest];
        array[largest] = temp;

        heapify(array, n, largest);
    }
}

void heapSort(int* array, int n) 
{
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(array, n, i);

    for (int i = n - 1; i > 0; i--) 
    {
        int temp = array[0];
        array[0] = array[i];
        array[i] = temp;

        heapify(array, i, 0);
    }
}

void mergeSort(int* array, int left, int right) 
{
    if (left < right) 
    {
        int mid = (left + right) / 2;
        if (right - left > 1000) 
        {
            #pragma omp taskgroup
            {
                #pragma omp task
                mergeSort(array, left, mid);
                #pragma omp task
                mergeSort(array, mid + 1, right);
            }
        } 
        else 
        {
            heapSort(array + left, mid - left + 1); 
            heapSort(array + mid + 1, right - mid); 
        }
        #pragma omp taskwait
        merge(array, left, mid, right);
    }
}

void parallelMergeSort(int* array, int n, int num_threads) 
{
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(array, 0, n - 1);
    }
}

int compare(const void * x1, const void * x2) 
{
    return (*(int*)x1 - *(int*)x2);
}

int are_equal(int* array1, int* array2, int N) 
{
    for (int i = 0; i < N; i++)
        if (array1[i] != array2[i])
            return 0;
    return 1;
}

int main(int argc, char *argv[])
{
    int N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    int* array1 = (int*)malloc(N * sizeof(int));
    int* array2 = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) 
    {
        array1[i] = array2[i] = rand() % 100; 
    }

    double start1 = omp_get_wtime();
    parallelMergeSort(array1, N, num_threads);
    double end1 = omp_get_wtime();
    double time1 = end1 - start1;
    
    double start2 = omp_get_wtime();
    qsort(array2, N, sizeof(int), compare);
    double end2 = omp_get_wtime();
    double time2 = end2 - start2;

    printf("Time of the parallel sort = %f sec\n", time1);
    printf("Time of qsort = %.7f sec\n", time2);

    if (are_equal(array1, array2, N))
    {
        printf("The sorted arrays are equal\n");
    }
    else printf("The sorted arrays are not equal\n");

    free(array1);
    free(array2);
    return 0;
}