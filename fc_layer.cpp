#include <stdio.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// 스레드 작업 정보 구조체
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

// 단일 스레드 작업 함수
void* fc_layer_thread(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    
    const size_t N = td->input_dim;      // 4096
    const size_t M = td->output_dim;     // 4096
    
    // 각 입력 인스턴스 처리
    for (size_t b = 0; b < td->data_cnt; b++) {
        float* input_base = td->input + b * N;
        float* output_base = td->output + b * M;
        
        // 이 스레드가 담당하는 출력 범위
        for (size_t m = td->start_idx; m < td->end_idx; m++) {
            
            // AVX2: 8개 float 동시 처리
            __m256 sum_vec = _mm256_setzero_ps();
            
            size_t n;
            // 메인 루프: 8개씩 처리
            for (n = 0; n + 7 < N; n += 8) {
                // 입력 8개 로드
                __m256 input_vec = _mm256_loadu_ps(&input_base[n]);
                
                // 가중치 8개 로드
                __m256 weight_vec = _mm256_loadu_ps(&td->matrix[n * M + m]);
                
                // FMA: sum = sum + (input * weight)
                sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
            }
            
            // 8개 부분합을 배열로 저장 후 합산 (가장 호환성 높음!)
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float sum = temp[0] + temp[1] + temp[2] + temp[3] + 
                       temp[4] + temp[5] + temp[6] + temp[7];
            
            // 나머지 처리 (N이 8로 나누어떨어지지 않는 경우)
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
    // 스레드 수 조정 (최소 1, 최대 출력 차원)
    if (threads < 1) threads = 1;
    if ((size_t)threads > output_dim) threads = output_dim;
    
    // 단일 스레드면 바로 실행
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
    
    // 멀티스레드 실행
    pthread_t* thread_handles = (pthread_t*)malloc(threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(threads * sizeof(ThreadData));
    
    // 출력 차원을 스레드 수로 분할
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
        
        // 나머지를 앞쪽 스레드에 분배
        size_t current_chunk = chunk_size + (t < (int)remainder ? 1 : 0);
        thread_data[t].end_idx = current_start + current_chunk;
        current_start += current_chunk;
        
        pthread_create(&thread_handles[t], NULL, fc_layer_thread, &thread_data[t]);
    }
    
    // 모든 스레드 종료 대기
    for (int t = 0; t < threads; t++) {
        pthread_join(thread_handles[t], NULL);
    }
    
    free(thread_handles);
    free(thread_data);
}