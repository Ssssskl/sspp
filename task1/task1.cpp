#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <chrono>

struct args_pth {
    double a, b, i, r;
};

double F(double x)
{
    return 4.0 / (1 + x*x);
}

void * Rectangle_method(void *tmp)
{
    double a, b, i, sum = 0.0;
    struct args_pth * argum = (struct args_pth *) tmp;
    a = argum -> a;
    b = argum -> b;
    i = argum -> i;

    while (a < b) {
        sum += F(a);
        a += i;
    }

    argum -> r = sum*i;
    return NULL;
}

int main(int argc, char** argv)
{   
    auto start = std::chrono::steady_clock::now();
    long int n = strtol(argv[1], NULL, 10), k = strtol(argv[2], NULL, 10);
    double a = 0.0, b = 1.0, sum = 0.0;
    double i1 = (b - a)/n;
    double i2 = (b - a)/k;

    pthread_t my_threads[k]; 
    struct args_pth args[k];

    for (int j = 0; j < k; j++) {
        args[j].a = a;
        a += i2;
        args[j].b = a;
        args[j].i = i1;
        pthread_create(&my_threads[j], NULL, Rectangle_method, &args[j]); 
    }
    
    for (int j = 0; j < k; j++) {
        pthread_join(my_threads[j], NULL);
        sum += args[j].r;
    }

    printf("%f\n", sum);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("Elapsed time: %lf sec\n", elapsed.count()/ 1.0E6);
}

 
  