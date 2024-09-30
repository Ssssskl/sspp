#include "pthread.h"
#include <iostream>
#include <queue>
#include <cassert>

using namespace std;

int num_active_prod = 0;

class MyConcurrentQueue {
    public:
        void reserve (int n) {
            queue_limit = n;
        }

        void put (int value) 
        {
            pthread_mutex_lock(&m);
            while (queue_q.size() == queue_limit)
            {
                pthread_cond_wait(&cget, &m);
            }
            queue_q.push(value);
            pthread_cond_broadcast(&cput);
            pthread_mutex_unlock(&m);
        }

        int get() 
        {
            pthread_mutex_lock(&m);
            while (queue_q.empty() && !done)
            {
                pthread_cond_wait(&cput, &m);
            }
            if (!queue_q.empty())
            {
                int value = queue_q.front();
                queue_q.pop();
                pthread_cond_broadcast(&cget);
                pthread_mutex_unlock(&m);
                return value;
            }
            else
            {
                pthread_mutex_unlock(&m);
                return -1000;
            }
        }

        void is_done()
        {
            pthread_mutex_lock(&m);
            done = true;
            pthread_cond_broadcast(&cput);
            pthread_mutex_unlock(&m);
        }

    private:
        long unsigned int queue_limit;
        queue<long unsigned int> queue_q;
        bool done = false;
        pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
        pthread_cond_t cput = PTHREAD_COND_INITIALIZER;
        pthread_cond_t cget = PTHREAD_COND_INITIALIZER;
        
};


MyConcurrentQueue my_queue;
pthread_mutex_t m_out = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m_active = PTHREAD_MUTEX_INITIALIZER;


void* producer_func (void* params) {
    int * arg = (int *) params;
    for (int i = 0; i < 1000; i++) {
        my_queue.put(i); 
        pthread_mutex_lock(&m_out);
        cout<<"Thread number "<<*arg<<" puts element "<<i<<endl;
        pthread_mutex_unlock(&m_out);
    }
    pthread_mutex_lock(&m_active);
    num_active_prod--;
    if (num_active_prod == 0) my_queue.is_done();
    pthread_mutex_unlock(&m_active);
    return NULL;

}

void *consumer_func(void *params) {
    int * arg = (int *) params;
    while(true){
        int a = my_queue.get();
        if(a == -1000) break;
        pthread_mutex_lock(&m_out);
        cout<<"Thread number "<<*arg<<" gets element "<<a<<endl;
        pthread_mutex_unlock(&m_out);
    }
    return NULL;

}


int main () {

    my_queue.reserve(100); 
    int N = 50;
    int args[N];
    for (int i = 1; i <= N; i++)
    {
        args[i] = i;
    }

    // // Базовый тест {1 - 1}

    // pthread_t thread_producer;
    // pthread_t thread_consumer;
    // num_active_prod = 1;

    // pthread_create(&thread_producer, NULL, &producer_func, &args[0]); 
    // pthread_create(&thread_consumer, NULL, &consumer_func, &args[0]); 

    // pthread_join(thread_producer, NULL);
    // pthread_join(thread_consumer, NULL);    

    // // {1 - N}

    pthread_t thread_producer;
    pthread_t consumer_threads[N]; 
    num_active_prod = 1;

    pthread_create(&thread_producer, NULL, &producer_func, &args[0]);

    for (int i = 0; i < N; i++)
    {
        pthread_create(&consumer_threads[i], NULL, &consumer_func, &args[i]); 
    }

    pthread_join(thread_producer, NULL);
    for (int j = 0; j < N; j++) {
        pthread_join(consumer_threads[j], NULL);
    }

    // // {N - 1}

    // pthread_t producer_threads[N]; 
    // pthread_t thread_consumer;
    // num_active_prod = N;

    // for (int i = 0; i < N; i++)
    // {
    //     pthread_create(&producer_threads[i], NULL, &producer_func, &args[i]); 
    // }
    // pthread_create(&thread_consumer, NULL, &consumer_func, &args[0]); 

    // for (int j = 0; j < N; j++) {
    //     pthread_join(producer_threads[j], NULL);
    // }
    // pthread_join(thread_consumer, NULL);

    // // {N - M}
    
    // int M = 100;
    // int args2[M];
    // for (int i = 1; i <= M; i++)
    // {
    //     args[i] = i;
    // }
    // pthread_t producer_threads[N]; 
    // pthread_t consumer_threads[M];
    // num_active_prod = N;

    // for (int i = 0; i < N; i++)
    // {
    //     pthread_create(&producer_threads[i], NULL, &producer_func, &args[i]); 
    //     pthread_create(&consumer_threads[i], NULL, &consumer_func, &args2[i]); 
    // }

    // for (int j = 0; j < N; j++) {
    //     pthread_join(producer_threads[j], NULL);
    //     pthread_join(consumer_threads[j], NULL);
    // }

}
