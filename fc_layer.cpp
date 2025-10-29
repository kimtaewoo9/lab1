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
    size_t start_batch;
    size_t end_batch;
    int thread_id;
} ThreadData;

// 효율적인 horizontal sum
static inline float hsum_ps_sse3(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

void* fc_layer_thread(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    const size_t N = td->input_dim;
    const size_t M = td->output_dim;
    
    // 각 배치 처리
    for (size_t b = td->start_batch; b < td->end_batch; b++) {
        float* input_base = td->input + b * N;
        float* output_base = td->output + b * M;
        
        // 모든 출력 처리
        for (size_t m = 0; m < M; m++) {
            
            __m256 sum_vec = _mm256_setzero_ps();
            
            // AVX2로 8개씩 처리
            size_t n;
            for (n = 0; n + 7 < N; n += 8) {
                // 입력 8개 로드
                __m256 input_vec = _mm256_loadu_ps(&input_base[n]);
                
                // 가중치 8개 gather
                __m256i vindex = _mm256_setr_epi32(
                    (n+0) * M + m,
                    (n+1) * M + m,
                    (n+2) * M + m,
                    (n+3) * M + m,
                    (n+4) * M + m,
                    (n+5) * M + m,
                    (n+6) * M + m,
                    (n+7) * M + m
                );
                __m256 weight_vec = _mm256_i32gather_ps(td->matrix, vindex, 4);
                
                // FMA
                sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
            }
            
            // Reduction
            float sum = hsum_ps_sse3(sum_vec);
            
            // 나머지 스칼라 처리
            for (; n < N; n++) {
                sum += input_base[n] * td->matrix[n * M + m];
            }
            
            // Bias 추가
            sum += td->bias[m];
            
            // ReLU
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
    if ((size_t)threads > data_cnt) threads = data_cnt;
    
    if (threads == 1) {
        ThreadData td;
        td.data_cnt = data_cnt;
        td.input_dim = input_dim;
        td.output_dim = output_dim;
        td.matrix = matrix;
        td.bias = bias;
        td.input = input;
        td.output = output;
        td.start_batch = 0;
        td.end_batch = data_cnt;
        td.thread_id = 0;
        
        fc_layer_thread(&td);
        return;
    }
    
    pthread_t* thread_handles = (pthread_t*)malloc(threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(threads * sizeof(ThreadData));
    
    size_t batch_per_thread = data_cnt / threads;
    size_t remainder = data_cnt % threads;
    
    size_t current_start = 0;
    
    for (int t = 0; t < threads; t++) {
        thread_data[t].data_cnt = data_cnt;
        thread_data[t].input_dim = input_dim;
        thread_data[t].output_dim = output_dim;
        thread_data[t].matrix = matrix;
        thread_data[t].bias = bias;
        thread_data[t].input = input;
        thread_data[t].output = output;
        thread_data[t].start_batch = current_start;
        thread_data[t].thread_id = t;
        
        size_t current_chunk = batch_per_thread + (t < (int)remainder ? 1 : 0);
        thread_data[t].end_batch = current_start + current_chunk;
        current_start += current_chunk;
        
        pthread_create(&thread_handles[t], NULL, fc_layer_thread, &thread_data[t]);
    }
    
    for (int t = 0; t < threads; t++) {
        pthread_join(thread_handles[t], NULL);
    }
    
    free(thread_handles);
    free(thread_data);
}