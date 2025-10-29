#include <stdio.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t data_cnt;
    size_t input_dim;
    size_t output_dim;
    float* matrix;
    float* bias;
    float* input;
    float* output;
    size_t start_idx;
    size_t end_idx;
    int thread_id;
} ThreadData;

void* fc_layer_thread(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    const size_t N = td->input_dim;
    const size_t M = td->output_dim;
    
    for (size_t b = 0; b < td->data_cnt; b++) {
        float* input_base = td->input + b * N;
        float* output_base = td->output + b * M;
        
        for (size_t m = td->start_idx; m < td->end_idx; m++) {
            
            __m256 sum_vec = _mm256_setzero_ps();
            
            size_t n;
            for (n = 0; n + 7 < N; n += 8) {
                __m256 input_vec = _mm256_loadu_ps(&input_base[n]);
                
                // 가중치도 연속된 메모리에서 로드
                __m256 weight_vec = _mm256_set_ps(
                    td->matrix[(n+7) * M + m],
                    td->matrix[(n+6) * M + m],
                    td->matrix[(n+5) * M + m],
                    td->matrix[(n+4) * M + m],
                    td->matrix[(n+3) * M + m],
                    td->matrix[(n+2) * M + m],
                    td->matrix[(n+1) * M + m],
                    td->matrix[(n+0) * M + m]
                );
                
                sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
            }
            
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            
            for (; n < N; n++) {
                sum += input_base[n] * td->matrix[n * M + m];
            }
            
            sum += td->bias[m];
            if (sum < 0.0f) sum = 0.0f;
            
            output_base[m] = sum;
        }
    }
    
    return NULL;
}

void fc_layer(
    size_t data_cnt, 
    size_t input_dim, 
    size_t output_dim, 
    float* matrix, 
    float* bias, 
    float* input, 
    float* output, 
    int threads
) {
    if (threads < 1) threads = 1;
    if ((size_t)threads > output_dim) threads = output_dim;
    
    if (threads == 1) {
        ThreadData td;
        td.data_cnt = data_cnt;
        td.input_dim = input_dim;
        td.output_dim = output_dim;
        td.matrix = matrix;
        td.bias = bias;
        td.input = input;
        td.output = output;
        td.start_idx = 0;
        td.end_idx = output_dim;
        td.thread_id = 0;
        
        fc_layer_thread(&td);
        return;
    }
    
    pthread_t* thread_handles = (pthread_t*)malloc(threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(threads * sizeof(ThreadData));
    
    size_t chunk_size = output_dim / threads;
    size_t remainder = output_dim % threads;
    
    size_t current_start = 0;
    
    for (int t = 0; t < threads; t++) {
        thread_data[t].data_cnt = data_cnt;
        thread_data[t].input_dim = input_dim;
        thread_data[t].output_dim = output_dim;
        thread_data[t].matrix = matrix;
        thread_data[t].bias = bias;
        thread_data[t].input = input;
        thread_data[t].output = output;
        thread_data[t].start_idx = current_start;
        thread_data[t].thread_id = t;
        
        size_t current_chunk = chunk_size + (t < (int)remainder ? 1 : 0);
        thread_data[t].end_idx = current_start + current_chunk;
        current_start += current_chunk;
        
        pthread_create(&thread_handles[t], NULL, fc_layer_thread, &thread_data[t]);
    }
    
    for (int t = 0; t < threads; t++) {
        pthread_join(thread_handles[t], NULL);
    }
    
    free(thread_handles);
    free(thread_data);
}