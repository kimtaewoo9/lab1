/*
 * fc_layer.cpp
 * * 이 파일은 Lab 1: Optimized Neural Network Software 과제를 위해
 * 최적화된 fc_layer 함수를 포함합니다.
 *
 * 최적화 전략:
 * 1. OpenMP를 사용한 병렬 처리: 가장 바깥쪽 루프(입력 배치 루프)를 병렬화합니다.
 * 2. 루프 순서 변경: 캐시 효율성을 극대화하기 위해 (j, i) 루프를 (i, j) 순서로 변경.
 * - Naive (j, i): A[i*M + j] -> Column-wise 접근 (캐시 미스율 높음)
 * - Optimized (i, j): A[i*M + j] -> Row-wise 접근 (스트리밍, 캐시 효율 높음)
 * 3. AVX2 SIMD 벡터화: 'AVX2 지원' 힌트를 활용, 256비트(float 8개) 단위로 연산. [cite: 473, 614]
 * 4. FMA (Fused Multiply-Add): _mm256_fmadd_ps를 사용해 곱셈과 덧셈을 단일 명령어로 처리. 
 * 5. Aligned Memory 활용: '64-byte aligned' 보장을 활용해 정렬된 load/store 사용. [cite: 442, 632]
 * 6. Branchless ReLU: _mm256_max_ps를 사용해 'if'문 없는 ReLU 구현. [cite: 194]
 */
#include "fc_layer.h"
#include <algorithm>   // std::max
#include <omp.h>       // OpenMP 헤더 
#include <immintrin.h> // AVX/AVX2/FMA 인트린직 헤더 [cite: 605]

void fc_layer(
    const float *X,  // 입력 피처 맵 (num_inputs * N)
    const float *A,  // 가중치 행렬 (N * M)
    const float *B,  // 편향 벡터 (M)
    float *Y,        // 출력 피처 맵 (num_inputs * M)
    int num_inputs,
    int N,
    int M,
    int threads      // CLI에서 전달된 스레드 수
) {
    // 1. OpenMP 스레드 수 설정
    // 과제에서 'threads' 인자를 전달하며, 이를 활용해 스레드를 생성해야 함 
    omp_set_num_threads(threads);

    // AVX2는 256비트 = 32바이트 = float 8개를 한 번에 처리 
    const int VEC_WIDTH = 8;
    
    // ReLU(max(0, x)) 연산을 위한 0으로 채워진 AVX 벡터
    const __m256 zero_vec = _mm256_setzero_ps(); // [cite: 628]

    // 2. OpenMP를 사용해 가장 바깥쪽 루프 (배치 루프)를 병렬화
    // 각 입력은 서로 독립적이므로 병렬화에 가장 적합
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < num_inputs; ++b) {
        
        // 현재 입력 및 출력에 대한 포인터
        const float* current_X = X + b * N;
        float* current_Y = Y + b * M;

        // --- 1단계: Y 벡터를 Bias(B) 값으로 초기화 (Y = B) ---
        // AVX2를 사용하여 8개씩 병렬 복사
        for (int j = 0; j < M; j += VEC_WIDTH) {
            // '64-byte aligned' 가 보장되므로 aligned load/store 사용 [cite: 442, 632]
            __m256 b_vec = _mm256_load_ps(&B[j]);
            _mm256_store_ps(&current_Y[j], b_vec);
        }

        // --- 2단계: 행렬-벡터 곱셈 (Y += X * A) ---
        // 캐시 효율을 위해 (i, j) 루프 순서로 실행 (Y[j] += X[i] * A[i][j])
        // A[i*M+j] 접근은 row-major  순서와 일치하여 캐시 친화적
        for (int i = 0; i < N; ++i) {
            // X[i] 값을 AVX 벡터의 8개 모든 레인에 복제 (broadcast)
            // [x, x, x, x, x, x, x, x]
            const __m256 x_vec = _mm256_set1_ps(current_X[i]);

            // A의 i번째 행을 8개씩 벡터화하여 Y에 누적
            for (int j = 0; j < M; j += VEC_WIDTH) {
                // Y[j..j+7] 로드
                __m256 y_vec = _mm256_load_ps(&current_Y[j]);
                // A[i*M + j ... j+7] 로드
                __m256 a_vec = _mm256_load_ps(&A[i * M + j]);

                // Fused Multiply-Add: Y = (X * A) + Y
                // y_vec = (x_vec * a_vec) + y_vec [cite: 612, 621, 647]
                y_vec = _mm256_fmadd_ps(x_vec, a_vec, y_vec);

                // 결과를 Y에 다시 저장
                _mm256_store_ps(&current_Y[j], y_vec);
            }
        }

        // --- 3단계: ReLU 활성화 함수 적용 (Y = max(0, Y)) ---
        // if문(분기) 대신 _mm256_max_ps를 사용 (Branchless) [cite: 194, 645]
        for (int j = 0; j < M; j += VEC_WIDTH) {
            __m256 y_vec = _mm256_load_ps(&current_Y[j]);
            // y_vec와 zero_vec 중 큰 값을 선택
            y_vec = _mm256_max_ps(y_vec, zero_vec);
            _mm256_store_ps(&current_Y[j], y_vec);
        }
    }
}