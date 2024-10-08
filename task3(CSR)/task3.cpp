#include <iostream>
#include <vector>
#include "papi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

class CSR_graph {
    int row_count; //number of vertices in graph
    unsigned int col_count; //number of edges in graph
    
    std::vector<unsigned int> row_ptr;
    std::vector<int> col_ids;
    std::vector<double> vals;

public:

    void read(const char* filename) {
        FILE *graph_file = fopen(filename, "rb");
        fread(reinterpret_cast<char*>(&row_count), sizeof(int), 1, graph_file);
        fread(reinterpret_cast<char*>(&col_count), sizeof(unsigned int), 1, graph_file);

        std::cout << "Row_count = " << row_count << ", col_count = " << col_count << std::endl;
        
        row_ptr.resize(row_count + 1);
        col_ids.resize(col_count);
        vals.resize(col_count);
         
        fread(reinterpret_cast<char*>(row_ptr.data()), sizeof(unsigned int), row_count + 1, graph_file);
        fread(reinterpret_cast<char*>(col_ids.data()), sizeof(int), col_count, graph_file);
        fread(reinterpret_cast<char*>(vals.data()), sizeof(double), col_count, graph_file);
        fclose(graph_file);
    }

    std::vector<unsigned int> getRow ()
    {
        return row_ptr;
    }

    std::vector<int> getCol()
    {
        return col_ids;
    }

    std::vector<double> getVal()
    {
        return vals;
    }


    void print_vertex(int idx) {
        for (int col = row_ptr[idx]; col < row_ptr[idx + 1]; col++) {
            std::cout << col_ids[col] << " " << vals[col] <<std::endl;
        }
        std::cout << std::endl;
    }

    void reset() {
        row_count = 0;
        col_count = 0;
        row_ptr.clear();
        col_ids.clear();
        vals.clear();
    }
}; 


#define N_TESTS 5

int main () {
    const char* filenames[N_TESTS];
    filenames[0] = "synt";
    filenames[1] = "road_graph";
    filenames[2] = "stanford";
    filenames[3] = "youtube";
    filenames[4] = "syn_rmat";
    

   for (int n_test = 0; n_test < N_TESTS; n_test++) {
        std::cout << std::endl << "---------  Test " << n_test + 1 << " ---------" << std::endl << std::endl;
        CSR_graph a;
        a.read(filenames[n_test]);
        long long values[3];
        int events[3] = {PAPI_L1_DCA, PAPI_L2_DCM, 0};
        int event_set = PAPI_NULL, event_code;

        PAPI_library_init(PAPI_VER_CURRENT);

        PAPI_create_eventset(&event_set);
        PAPI_event_name_to_code("perf::PERF_COUNT_HW_CACHE_MISSES", &event_code);
        events[2] = event_code;
        PAPI_add_events(event_set, events, 3);

        PAPI_start(event_set);

        double cnt, maxim = 0; 
        int maxid = 0;
        std::vector<unsigned int> row_ptr = a.getRow();
        std::vector<int> col_ids = a.getCol();
        std::vector<double> vals = a.getVal();
        for (int i = 0 ; i < row_ptr.size() - 1; i++)
        {
            cnt = 0;
            for (int j = row_ptr[i]; j < row_ptr[i+1]; j++)
            {
                if (col_ids[j] % 2 == 0)
                {
                    cnt += vals[j];
                }
            }
            if (cnt > maxim)
            {
                maxid = i;
                maxim = cnt;
            } 
        }
        std::cout<<"alg1   "<<maxid<<std::endl;

        PAPI_stop(event_set, values);

        std::cout << "L1_DCA: " << values[0] << std::endl;
        std::cout << "L2_DCM: " << values[1] << std::endl;
        std::cout << "COUNT_HW_CACHE_MISSES: " << values[2]<<std::endl<<std::endl;

        PAPI_cleanup_eventset(event_set);
        PAPI_destroy_eventset(&event_set);

        double cnt2, maxim2 = 0; 
        int maxid2 = 0;

        long long values2[3];
        int events2[3] = {PAPI_L1_DCA, PAPI_L2_DCM, 0};
        int event_set2 = PAPI_NULL, event_code2;

        PAPI_library_init(PAPI_VER_CURRENT);

        PAPI_create_eventset(&event_set2);
        PAPI_event_name_to_code("perf::PERF_COUNT_HW_CACHE_MISSES", &event_code2);
        events2[2] = event_code2;
        PAPI_add_events(event_set2, events2, 3);

        PAPI_start(event_set2);

        for (int i = 0 ; i < row_ptr.size() - 1; i++)
        {
            cnt2 = 0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            {
                int ver = col_ids[j];
                double w = 0;
                for (int k = row_ptr[ver]; k < row_ptr[ver + 1]; k++)
                {
                    int ver2 = col_ids[k];
                    w += vals[k]*(row_ptr[ver2 + 1] - row_ptr[ver2]);
                }
                cnt2+= vals[j]* w;
            }
            if (cnt2 > maxim2)
            {
                maxid2 = i;
                maxim2 = cnt2;
            }
        }
        std::cout<<"alg2        "<<maxid2<<std::endl;

        PAPI_stop(event_set2, values2);

        std::cout << "L1_DCA: " << values2[0] << std::endl;
        std::cout << "L2_DCM: " << values2[1] << std::endl;
        std::cout << "COUNT_HW_CACHE_MISSES: " << values2[2]<<std::endl<<std::endl;

        PAPI_cleanup_eventset(event_set2);
        PAPI_destroy_eventset(&event_set2);

        a.reset();
    }
}