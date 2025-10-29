/*
 * fc_layer.cpp (최종 수정본)
 *
 * main.cpp의 'extern' 선언과 정확히 일치하는 함수 시그니처를 사용합니다.
 * 별도의 .h 파일 없이 이 파일만으로 컴파일이 가능합니다.
 *
 * 모든 최적화 전략 (OpenMP, AVX2, FMA, 캐시 최적화, Branchless)이
 * 올바른 변수명(data_cnt, input_dim 등)으로 적용되어 있습니다.
 */
#include <stdlib.h>    // size_t 자료형 정의
#include <omp.h>       // OpenMP 헤더
#include <immintrin.h> // AVX/AVX2/FMA 인트린직 헤더

// main.cpp에서 extern으로 선언한 함수 시그니처와 정확히 일치
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
    // --- 최적화 로직의 가독성을 위해 main.cpp의 변수명을 별칭(alias)으로 사용 ---
    // (컴파일 시 이 부분은 최적화되어 사라짐)
    const float *X = input;
    const float *A = matrix;
    const float *B = bias;
    float *Y = output;
    
    // OpenMP 루프 등에서 사용하기 위해 size_t를 int로 변환
    const int num_inputs = (int)data_cnt;
    const int N = (int)input_dim;
    const int M = (int)output_dim;
    // --- 별칭 설정 끝 ---


    [cite_start]// 1. OpenMP 스레드 수 설정 (과제 요구사항) [cite: 729]
    omp_set_num_threads(threads);

    [cite_start]// AVX2는 256비트 = 32바이트 = float 8개를 한 번에 처리 [cite: 80, 140]
    const int VEC_WIDTH = 8;
    
    [cite_start]// ReLU(max(0, x)) 연산을 위한 0으로 채워진 AVX 벡터 (Branchless 구현용) [cite: 154]
    const __m256 zero_vec = _mm256_setzero_ps();

    // 2. OpenMP로 가장 바깥쪽 루프 (배치 루프, b) 병렬화
    // 각 입력 배치는(b)는 서로 독립적이므로 병렬화에 가장 적합
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < num_inputs; ++b) {
        
        // 포인터 계산 시에는 오버플로우 방지를 위해 원본 size_t 타입 사용
        const float* current_X = X + (size_t)b * N;
        float* current_Y = Y + (size_t)b * M;

        // --- 1단계: Y 벡터를 Bias(B) 값으로 초기화 (Y = B) ---
        // AVX를 사용해 8개씩 병렬 복사
        for (int j = 0; j < M; j += VEC_WIDTH) {
            [cite_start]// '64-byte aligned' 보장을 활용해 aligned load 사용 [cite: 728]
            __m256 b_vec = _mm256_load_ps(&B[j]);
            _mm256_store_ps(&current_Y[j], b_vec);
        }

        // --- 2단계: 행렬-벡터 곱셈 (Y += X * A) ---
        // 캐시 효율을 위한 (i, j) 루프 순서 (가중치 A를 행 순서로 순차 접근)
        for (int i = 0; i < N; ++i) {
            // X[i] 값을 AVX 벡터 8개 레인에 복제 (broadcast)
            const __m256 x_vec = _mm256_set1_ps(current_X[i]);

            // A의 i번째 행을 8개씩(j 루프) 벡터화하여 Y에 누적
            for (int j = 0; j < M; j += VEC_WIDTH) {
                // Y[j..j+7] 로드 (aligned)
                __m256 y_vec = _mm256_load_ps(&current_Y[j]);
                // 가중치 행렬 A 접근 시 오버플로우 방지 (size_t 사용)
                __m256 a_vec = _mm256_load_ps(&A[(size_t)i * M + j]);

                [cite_start]// Fused Multiply-Add (핵심 연산): Y = (X * A) + Y [cite: 138, 146]
                y_vec = _mm256_fmadd_ps(x_vec, a_vec, y_vec);
                
                // 결과를 Y에 다시 저장 (aligned)
                _mm256_store_ps(&current_Y[j], y_vec);
            }
        }

        // --- 3단계: ReLU 활성화 함수 적용 (Y = max(0, Y)) ---
        // if문(분기) 대신 _mm256_max_ps를 사용 (Branchless)
        for (int j = 0; j < M; j += VEC_WIDTH) {
            __m256 y_vec = _mm256_load_ps(&current_Y[j]);
            // y_vec와 zero_vec(0.0f) 중 큰 값을 선택
            y_vec = _mm256_max_ps(y_vec, zero_vec);
            _mm256_store_ps(&current_Y[j], y_vec);
        }
    }
}