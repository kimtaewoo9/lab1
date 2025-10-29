#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h> // AVX/AVX2/FMA 인트린직 헤더

// 스레드에 작업을 전달하기 위한 구조체
typedef struct {
    const float *X;
    const float *A;
    const float *B;
    float *Y;
    int N; // input_dim
    int M; // output_dim
    int start_b; // 이 스레드가 처리할 시작 배치(batch) 인덱스
    int end_b;   // 이 스레드가 처리할 끝 배치(batch) 인덱스
} ThreadData;


// 각 스레드가 실행할 함수
void* fc_layer_thread(void* arg) {
    ThreadData* td = (ThreadData*)arg;

    const float *X = td->X;
    const float *A = td->A;
    const float *B = td->B;
    float *Y = td->Y;
    const int N = td->N;
    const int M = td->M;
    
    const int VEC_WIDTH = 8;
    const __m256 zero_vec = _mm256_setzero_ps();

    // 이 스레드에 할당된 배치(batch) 범위만큼만 루프 실행
    for (int b = td->start_b; b < td->end_b; ++b) {
        
        const float* current_X = X + (size_t)b * N;
        float* current_Y = Y + (size_t)b * M;

        for (int j = 0; j < M; j += VEC_WIDTH) {
            __m256 b_vec = _mm256_load_ps(&B[j]);
            _mm256_store_ps(&current_Y[j], b_vec);
        }

        // 캐시 효율을 위한 (i, j) 루프 순서        
        int i = 0;
        // i 루프를 4개씩 언롤링 
        for (; i + 3 < N; i += 4) {
            // X값 4개 미리 로드
            const __m256 x_vec0 = _mm256_set1_ps(current_X[i]);
            const __m256 x_vec1 = _mm256_set1_ps(current_X[i + 1]);
            const __m256 x_vec2 = _mm256_set1_ps(current_X[i + 2]);
            const __m256 x_vec3 = _mm256_set1_ps(current_X[i + 3]);

            // A의 4개 행(row)을 8개씩(j 루프) 벡터화하여 Y에 누적
            for (int j = 0; j < M; j += VEC_WIDTH) {
                // Y를 1번 Load
                __m256 y_vec = _mm256_load_ps(&current_Y[j]);

                // A값 4개 로드
                __m256 a_vec0 = _mm256_load_ps(&A[((size_t)i) * M + j]);
                __m256 a_vec1 = _mm256_load_ps(&A[((size_t)i + 1) * M + j]);
                __m256 a_vec2 = _mm256_load_ps(&A[((size_t)i + 2) * M + j]);
                __m256 a_vec3 = _mm256_load_ps(&A[((size_t)i + 3) * M + j]);

                // FMA 4번 수행
                y_vec = _mm256_fmadd_ps(x_vec0, a_vec0, y_vec);
                y_vec = _mm256_fmadd_ps(x_vec1, a_vec1, y_vec);
                y_vec = _mm256_fmadd_ps(x_vec2, a_vec2, y_vec);
                y_vec = _mm256_fmadd_ps(x_vec3, a_vec3, y_vec);
                
                // Y를 1번 Store
                _mm256_store_ps(&current_Y[j], y_vec);
            }
        }
        
        // 나머지 처리 루프 (N이 4의 배수가 아닐 경우. 4096은 4의 배수라 실행 안 됨)
        for (; i < N; ++i) {
            const __m256 x_vec = _mm256_set1_ps(current_X[i]);
            for (int j = 0; j < M; j += VEC_WIDTH) {
                __m256 y_vec = _mm256_load_ps(&current_Y[j]);
                __m256 a_vec = _mm256_load_ps(&A[(size_t)i * M + j]);
                y_vec = _mm256_fmadd_ps(x_vec, a_vec, y_vec);
                _mm256_store_ps(&current_Y[j], y_vec);
            }
        }

        for (int j = 0; j < M; j += VEC_WIDTH) {
            __m256 y_vec = _mm256_load_ps(&current_Y[j]);
            y_vec = _mm256_max_ps(y_vec, zero_vec);
            _mm256_store_ps(&current_Y[j], y_vec);
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
    pthread_t* thread_handles = (pthread_t*)malloc(threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(threads * sizeof(ThreadData));

    const int num_inputs = (int)data_cnt;

    int chunk_size = num_inputs / threads;
    int remainder = num_inputs % threads;
    int current_start_b = 0;

    for (int t = 0; t < threads; ++t) {
        thread_data[t].X = input;
        thread_data[t].A = matrix;
        thread_data[t].B = bias;
        thread_data[t].Y = output;
        thread_data[t].N = (int)input_dim;
        thread_data[t].M = (int)output_dim;

        thread_data[t].start_b = current_start_b;
        int current_chunk = chunk_size + (t < remainder ? 1 : 0);
        thread_data[t].end_b = current_start_b + current_chunk;
        current_start_b += current_chunk;

        pthread_create(&thread_handles[t], NULL, fc_layer_thread, &thread_data[t]);
    }

    for (int t = 0; t < threads; ++t) {
        pthread_join(thread_handles[t], NULL);
    }

    free(thread_handles);
    free(thread_data);
}