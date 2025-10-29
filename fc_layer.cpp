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
    size_t start_m;
    size_t end_m;
    int thread_id;
} ThreadData;

void* fc_layer_thread(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    const size_t N = td->input_dim;
    const size_t M = td->output_dim;
    
    // 각 배치 처리
    for (size_t b = 0; b < td->data_cnt; b++) {
        float* input_base = td->input + b * N;
        float* output_base = td->output + b * M;
        
        // 이 스레드가 담당하는 출력 범위
        for (size_t m = td->start_m; m < td->end_m; m++) {
            
            // 8개씩 unroll
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
            
            size_t n;
            // 32개씩 unroll (4 × 8)
            for (n = 0; n + 31 < N; n += 32) {
                // Block 0
                __m256 in0 = _mm256_loadu_ps(&input_base[n]);
                __m256 w0 = _mm256_set_ps(
                    td->matrix[(n+7)*M + m],
                    td->matrix[(n+6)*M + m],
                    td->matrix[(n+5)*M + m],
                    td->matrix[(n+4)*M + m],
                    td->matrix[(n+3)*M + m],
                    td->matrix[(n+2)*M + m],
                    td->matrix[(n+1)*M + m],
                    td->matrix[(n+0)*M + m]
                );
                sum0 = _mm256_fmadd_ps(in0, w0, sum0);
                
                // Block 1
                __m256 in1 = _mm256_loadu_ps(&input_base[n+8]);
                __m256 w1 = _mm256_set_ps(
                    td->matrix[(n+15)*M + m],
                    td->matrix[(n+14)*M + m],
                    td->matrix[(n+13)*M + m],
                    td->matrix[(n+12)*M + m],
                    td->matrix[(n+11)*M + m],
                    td->matrix[(n+10)*M + m],
                    td->matrix[(n+9)*M + m],
                    td->matrix[(n+8)*M + m]
                );
                sum1 = _mm256_fmadd_ps(in1, w1, sum1);
                
                // Block 2
                __m256 in2 = _mm256_loadu_ps(&input_base[n+16]);
                __m256 w2 = _mm256_set_ps(
                    td->matrix[(n+23)*M + m],
                    td->matrix[(n+22)*M + m],
                    td->matrix[(n+21)*M + m],
                    td->matrix[(n+20)*M + m],
                    td->matrix[(n+19)*M + m],
                    td->matrix[(n+18)*M + m],
                    td->matrix[(n+17)*M + m],
                    td->matrix[(n+16)*M + m]
                );
                sum2 = _mm256_fmadd_ps(in2, w2, sum2);
                
                // Block 3
                __m256 in3 = _mm256_loadu_ps(&input_base[n+24]);
                __m256 w3 = _mm256_set_ps(
                    td->matrix[(n+31)*M + m],
                    td->matrix[(n+30)*M + m],
                    td->matrix[(n+29)*M + m],
                    td->matrix[(n+28)*M + m],
                    td->matrix[(n+27)*M + m],
                    td->matrix[(n+26)*M + m],
                    td->matrix[(n+25)*M + m],
                    td->matrix[(n+24)*M + m]
                );
                sum3 = _mm256_fmadd_ps(in3, w3, sum3);
            }
            
            // 4개 합치기
            sum0 = _mm256_add_ps(sum0, sum1);
            sum2 = _mm256_add_ps(sum2, sum3);
            sum0 = _mm256_add_ps(sum0, sum2);
            
            // Horizontal sum
            float temp[8];
            _mm256_storeu_ps(temp, sum0);
            float sum = temp[0] + temp[1] + temp[2] + temp[3] + 
                       temp[4] + temp[5] + temp[6] + temp[7];
            
            // 나머지 처리
            for (; n < N; n++) {
                sum += input_base[n] * td->matrix[n * M + m];
            }
            
            // Bias + ReLU
            sum += td->bias[m];
            output_base[m] = (sum > 0.0f) ? sum : 0.0f;
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
        td.start_m = 0;
        td.end_m = output_dim;
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
        thread_data[t].start_m = current_start;
        thread_data[t].thread_id = t;
        
        size_t current_chunk = chunk_size + (t < (int)remainder ? 1 : 0);
        thread_data[t].end_m = current_start + current_chunk;
        current_start += current_chunk;
        
        pthread_create(&thread_handles[t], NULL, fc_layer_thread, &thread_data[t]);
    }
    
    for (int t = 0; t < threads; t++) {
        pthread_join(thread_handles[t], NULL);
    }
    
    free(thread_handles);
    free(thread_data);
}