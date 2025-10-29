/*
 * fc_layer.cpp (최종 수정본: pthread + AVX)
 *
 * Makefile 수정 없이 (-lpthread만 사용) 컴파일되도록 pthread로 재작성했습니다.
 * 과제에서 요구하는 모든 최적화 기법을 포함합니다:
 * 1. Pthread 병렬 처리: 'threads' 인자를 사용해 배치(data_cnt) 작업을 분배.
 * 2. 캐시 최적화 (루프 순서): (i, j) 순서로 루프를 변경해 가중치 행렬(A)을 순차 접근.
 * 3. AVX2 SIMD 벡터화: 출력(j) 루프를 8개씩(256비트) 묶어서 처리.
 * 4. FMA (Fused Multiply-Add): _mm256_fmadd_ps 사용.
 * 5. Aligned Memory 활용: 'load_ps' (aligned load) 사용.
 * 6. Branchless ReLU: _mm256_max_ps 사용.
 */
#include <stdlib.h>    // size_t, malloc, free
#include <pthread.h>   // pthread_t, pthread_create, pthread_join
#include <immintrin.h> // AVX/AVX2/FMA 인트린직 헤더

// 스레드에 작업을 전달하기 위한 구조체
typedef struct {
    // 공통 데이터 (모든 스레드가 공유)
    const float *X;
    const float *A;
    const float *B;
    float *Y;
    int N; // input_dim
    int M; // output_dim

    // 스레드별 작업 범위
    int start_b; // 이 스레드가 처리할 시작 배치(batch) 인덱스
    int end_b;   // 이 스레드가 처리할 끝 배치(batch) 인덱스

} ThreadData;


// 각 스레드가 실행할 함수
void* fc_layer_thread(void* arg) {
    ThreadData* td = (ThreadData*)arg;

    // 가독성을 위해 변수명 다시 설정
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
        
        // 포인터 계산
        const float* current_X = X + (size_t)b * N;
        float* current_Y = Y + (size_t)b * M;

        // --- 1단계: Y = B (Bias 초기화) ---
        for (int j = 0; j < M; j += VEC_WIDTH) {
            __m256 b_vec = _mm256_load_ps(&B[j]);
            _mm256_store_ps(&current_Y[j], b_vec);
        }

        // --- 2단계: Y += X * A (행렬 곱) ---
        // 캐시 효율을 위한 (i, j) 루프 순서
        for (int i = 0; i < N; ++i) {
            const __m256 x_vec = _mm256_set1_ps(current_X[i]);
            for (int j = 0; j < M; j += VEC_WIDTH) {
                __m256 y_vec = _mm256_load_ps(&current_Y[j]);
                __m256 a_vec = _mm256_load_ps(&A[(size_t)i * M + j]);

                // Fused Multiply-Add
                y_vec = _mm256_fmadd_ps(x_vec, a_vec, y_vec);
                _mm256_store_ps(&current_Y[j], y_vec);
            }
        }

        // --- 3단계: Y = ReLU(Y) (Branchless) ---
        for (int j = 0; j < M; j += VEC_WIDTH) {
            __m256 y_vec = _mm256_load_ps(&current_Y[j]);
            y_vec = _mm256_max_ps(y_vec, zero_vec);
            _mm256_store_ps(&current_Y[j], y_vec);
        }
    }
    
    return NULL;
}


// main.cpp에서 호출하는 메인 함수
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
    // 스레드 핸들 및 데이터 구조체 배열 동적 할당
    pthread_t* thread_handles = (pthread_t*)malloc(threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(threads * sizeof(ThreadData));

    const int num_inputs = (int)data_cnt;

    // 작업을 'threads' 개수만큼 분배
    int chunk_size = num_inputs / threads;
    int remainder = num_inputs % threads;
    int current_start_b = 0;

    for (int t = 0; t < threads; ++t) {
        // 스레드에 공통 데이터 설정
        thread_data[t].X = input;
        thread_data[t].A = matrix;
        thread_data[t].B = bias;
        thread_data[t].Y = output;
        thread_data[t].N = (int)input_dim;
        thread_data[t].M = (int)output_dim;

        // 스레드별 작업 범위(배치) 설정
        thread_data[t].start_b = current_start_b;
        int current_chunk = chunk_size + (t < remainder ? 1 : 0);
        thread_data[t].end_b = current_start_b + current_chunk;
        current_start_b += current_chunk;

        // 스레드 생성
        pthread_create(&thread_handles[t], NULL, fc_layer_thread, &thread_data[t]);
    }

    // 모든 스레드가 끝날 때까지 대기
    for (int t = 0; t < threads; ++t) {
        pthread_join(thread_handles[t], NULL);
    }

    // 할당된 메모리 해제
    free(thread_handles);
    free(thread_data);
}