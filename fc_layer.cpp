#include <stdio.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct {
    size_t data_cnt;
    size_t input_dim;
    size_t output_dim;
    float* matrix;
    float* bias;
    float* input;
    float* output;
    size_t start_n;
    size_t end_n;
} ThreadData;

void* fc_layer_thread(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    const size_t N = td->input_dim;
    const size_t M = td->output_dim;
    
    // Loop order: (b, n, m) - 캐시 친화적!
    for (size_t b = 0; b < td->data_cnt; b++) {
        
        // 이 스레드가 담당하는 n 범위
        for (size_t n = td->start_n; n < td->end_n; n++) {
            
            float input_val = td->input[b * N + n];
            __m256 input_vec = _mm256_set1_ps(input_val);  // Broadcast
            
            float* weight_row = td->matrix + n * M;  // 이 행의 시작
            
            // m을 8개씩 처리 (Row-wise 접근!)
            size_t m;
            for (m = 0; m + 7 < M; m += 8) {
                // 가중치 8개 연속 로드 (캐시 효율 최고!)
                __m256 weight_vec = _mm256_load_ps(&weight_row[m]);
                
                // 기존 출력 로드
                __m256 output_vec = _mm256_load_ps(&td->output[b * M + m]);
                
                // FMA: output += input * weight
                output_vec = _mm256_fmadd_ps(input_vec, weight_vec, output_vec);
                
                // 저장
                _mm256_store_ps(&td->output[b * M + m], output_vec);
            }
            
            // 나머지 스칼라 처리
            for (; m < M; m++) {
                td->output[b * M + m] += input_val * weight_row[m];
            }
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
    // 출력 초기화 (bias로)
    for (size_t b = 0; b < data_cnt; b++) {
        for (size_t m = 0; m < output_dim; m++) {
            output[b * output_dim + m] = bias[m];
        }
    }
    
    if (threads < 1) threads = 1;
    if ((size_t)threads > input_dim) threads = input_dim;
    
    if (threads == 1) {
        ThreadData td;
        td.data_cnt = data_cnt;
        td.input_dim = input_dim;
        td.output_dim = output_dim;
        td.matrix = matrix;
        td.bias = bias;
        td.input = input;
        td.output = output;
        td.start_n = 0;
        td.end_n = input_dim;
        
        fc_layer_thread(&td);
    } else {
        pthread_t* threads_arr = (pthread_t*)malloc(threads * sizeof(pthread_t));
        ThreadData* td_arr = (ThreadData*)malloc(threads * sizeof(ThreadData));
        
        size_t chunk = input_dim / threads;
        size_t remainder = input_dim % threads;
        size_t start = 0;
        
        for (int t = 0; t < threads; t++) {
            td_arr[t].data_cnt = data_cnt;
            td_arr[t].input_dim = input_dim;
            td_arr[t].output_dim = output_dim;
            td_arr[t].matrix = matrix;
            td_arr[t].bias = bias;
            td_arr[t].input = input;
            td_arr[t].output = output;
            td_arr[t].start_n = start;
            
            size_t my_chunk = chunk + (t < (int)remainder ? 1 : 0);
            td_arr[t].end_n = start + my_chunk;
            start += my_chunk;
            
            pthread_create(&threads_arr[t], NULL, fc_layer_thread, &td_arr[t]);
        }
        
        for (int t = 0; t < threads; t++) {
            pthread_join(threads_arr[t], NULL);
        }
        
        free(threads_arr);
        free(td_arr);
    }
    
    // ReLU 적용 (Branchless!)
    for (size_t b = 0; b < data_cnt; b++) {
        size_t m;
        __m256 zero = _mm256_setzero_ps();
        
        for (m = 0; m + 7 < output_dim; m += 8) {
            __m256 out_vec = _mm256_load_ps(&output[b * output_dim + m]);
            out_vec = _mm256_max_ps(out_vec, zero);  // Branchless ReLU!
            _mm256_store_ps(&output[b * output_dim + m], out_vec);
        }
        
        // 나머지
        for (; m < output_dim; m++) {
            float val = output[b * output_dim + m];
            output[b * output_dim + m] = (val > 0.0f) ? val : 0.0f;
        }
    }
}