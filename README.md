## 🚀 Lab 1: High-Performance NN Layer Optimization

Native C++로 작성된 VGG19 모델의 단일 Fully Connected (FC) 레이어를 하드웨어 한계까지 최적화하는 것을 목표로 합니다.

채점 기준인 4096-size 데이터셋과 8스레드 환경에서, Naïve 버전(236 MFLOPS) 대비 약 68배의 성능 향상을 달성했습니다.

## 💡 적용된 핵심 최적화 기법

1. 병렬화
2. SIMD 벡터화 (AVX2 & FMA)
3. 캐시 최적화 (루프 순서 변경)
4. Branchless ReLU
5. Loop Unrolling
6. 정렬된 메모리 접근
