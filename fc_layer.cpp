#include <stdio.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// 캐시 블로킹 타일 크기
#define TILE_M 64
#define TILE_N 256

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

// 8개 float를 horizontal add로 합산
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
    
    // 각 배치 처리 (스레드별로 배치 분할)
    for (size_t b = td->start_batch; b < td->end_batch; b++) {
        float* input_base = td->input + b * N;
        float* output_base = td->output + b * M;
        
        // 출력 차원에 대해 타일링
        for (size_t m_tile = 0; m_tile < M; m_tile += TILE_M) {
            size_t m_end = (m_tile + TILE_M < M) ? m_tile + TILE_M : M;
            
            // 입력 차원에 대해 타일링
            for (size_t n_tile = 0; n_tile < N; n_tile += TILE_N) {
                size_t n_end = (n_tile + TILE_N < N) ? n_tile + TILE_N : N;
                
                // 타일 내부 처리 (출력 차원)
                for (size_t m = m_tile; m < m_end; m++) {
                    
                    // 첫 타일이면 초기화, 아니면 누적
                    __m256 sum_vec = _mm256_setzero_ps();
                    
                    // AVX2 SIMD로 8개씩 처리
                    size_t n;
                    for (n = n_tile; n + 7 < n_end; n += 8) {
                        // 입력 8개 로드 (연속 메모리)
                        __m256 input_vec = _mm256_loadu_ps(&input_base[n]);
                        
                        // 가중치 8개 gather (비연속 메모리)
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
                        
                        // FMA: sum = sum + (input * weight)
                        sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
                    }
                    
                    // 8개 부분합 축약
                    float sum = hsum_ps_sse3(sum_vec);
                    
                    // 나머지 스칼라 처리
                    for (; n < n_end; n++) {
                        sum += input_base[n] * td->matrix[n * M + m];
                    }
                    
                    // 첫 타일이면 bias 추가 및 출력 저장
                    if (n_tile == 0) {
                        sum += td->bias[m];
                        // ReLU
                        output_base[m] = (sum > 0.0f) ? sum : 0.0f;
                    } else {
                        // 누적
                        output_base[m] += sum;
                    }
                }
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
    // 스레드 수 검증 및 조정
    if (threads < 1) threads = 1;
    if ((size_t)threads > data_cnt) threads = data_cnt;
    
    // 단일 스레드 처리
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
    
    // 멀티스레드 처리 (배치 차원으로 분할)
    pthread_t* thread_handles = (pthread_t*)malloc(threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(threads * sizeof(ThreadData));
    
    // 배치를 스레드 수로 분할
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
        
        // 나머지를 앞쪽 스레드에 분배
        size_t current_chunk = batch_per_thread + (t < (int)remainder ? 1 : 0);
        thread_data[t].end_batch = current_start + current_chunk;
        current_start += current_chunk;
        
        pthread_create(&thread_handles[t], NULL, fc_layer_thread, &thread_data[t]);
    }
    
    // 모든 스레드 완료 대기
    for (int t = 0; t < threads; t++) {
        pthread_join(thread_handles[t], NULL);
    }
    
    free(thread_handles);
    free(thread_data);
}