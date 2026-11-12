# Pruner_alpa

소형 자율주행 VLA(Alpamayo)에 복원 기반 시각 토큰 프루너를 적용하고, 프루닝이 소형 객체 정보에 주는 영향을 분석하는 연구 코드입니다.

> **현재 상태 (2026-11-12):** 마스킹 데이터 생성까지 끝났습니다. Pruner 학습 코드는 설계 단계이며 아직 작성하지 않았습니다. 학습 결과나 주행 평가 결과는 없습니다.

## 연구 질문

시각 토큰을 프루닝하면 신호등·작은 보행자·원거리 차량처럼 주행에 중요한 소형 객체의 정보가 충분히 보존되지 않을 수 있습니다. 이 정보를 담은 토큰을 복구하면 프루닝으로 늘어난 주행 오차가 완화되는지 확인합니다.

이는 아직 실험으로 확인하지 않은 가설입니다.

## 대상 모델과 입력

- **모델:** 연구실에서 구축한 Qwen3-VL(Cosmos-Reason2-2B) 기반 약 3~4B 규모의 소형 Alpamayo. 추론 환경은 Jetson Thor입니다.
- **입력:** 카메라 6대 × 4개 시점(0.1초 간격) = 이미지 24장. 각 이미지는 576×320이며 따로따로 모델에 들어갑니다.
- **시각 토큰:** 이미지 1장당 18×10 = 180토큰(토큰 1개 = 32×32픽셀), 입력 1회당 4,320토큰입니다.
- **데이터:** NVIDIA PhysicalAI-Autonomous-Vehicles에서 뽑은 입력 window. window 하나는 `(clip_id, t0_us)`로 구분합니다.

## FastDriveVLA와의 관계

[FastDriveVLA](https://arxiv.org/abs/2507.23318)의 ReconPruner 방식을 참고한 **재구현·확장**입니다. 공식 Pruner·복원 디코더 코드와 가중치는 찾지 못했고, 논문에 없는 세부 사항(복원 디코더 구조, 초기화, optimizer 등)은 직접 정했습니다.

모델, 데이터, 해상도가 모두 원 논문과 다릅니다. 따라서 이 코드의 결과를 FastDriveVLA 자체의 성능이나 결함으로 해석하면 안 됩니다.

## 진행 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | Grounded-SAM 전경 마스킹 데이터 생성 | 완료 |
| 2 | 복원 기반 Pruner 학습 | 설계 중 |
| 3 | 소형 객체 정보 손실·주행 오차 분석 | 예정 |
| 4 | 탈락 토큰 복구 실험 | 예정 |

## 1단계: 마스킹 데이터

Grounding DINO(`IDEA-Research/grounding-dino-base`)로 박스를 찾고 SAM 2.1(`facebook/sam2.1-hiera-large`)로 마스크를 만듭니다. 전경 클래스는 FastDriveVLA와 같이 사람·도로·차량·교통 표지와 신호등·교통 장벽입니다.

### 생성 결과

- window 2,784개, 이미지 66,816장 (Thor에서 장당 약 1.3초)
- 학습 후보는 평가 clip과 겹치지 않는 1,952 window(281 clip, 46,848장)입니다.
- 평가용은 328 window(328 clip)이며 학습에 쓰지 않습니다.

### 저장 구조

```
data/
  index.jsonl                  # window마다 한 줄: 출처, 소속 세트, maneuver
  exclude_noroad.jsonl         # 학습 손실에서 제외할 이미지 목록
  windows/<clip_id>_<t0_us>/
    cam{c}_f{f}.jpg            # 원본 (모델 입력과 같은 576×320)
    cam{c}_f{f}_fg.png         # 이진 전경 마스크
    cam{c}_f{f}_fore.jpg       # 전경 정답 = 원본 × 마스크
    cam{c}_f{f}_back.jpg       # 배경 정답 = 원본 × (1 − 마스크)
    cam{c}_f{f}_inst.npz       # 객체별 마스크
    meta.json                  # 카메라·타임스탬프, 객체별 클래스·score·박스·면적·토큰 칸
```

데이터는 PhysicalAI-AV 라이선스 때문에 이 저장소에 올리지 않습니다.

### 알려진 한계

- **도로 누락:** 이미지의 5.9%(3,960장)는 도로가 검출되지 않았습니다. 주로 야간·눈·안개 장면입니다. `exclude_noroad.jsonl`에 기록했고 학습 손실에서 제외합니다.
- **자차 차체:** 전방 카메라의 보닛이 일부 이미지에서 전경(도로 또는 차량)으로 잡힙니다. 따로 처리하지 않았습니다.
- **소형 객체:** 검출된 신호등·표지의 95%, 사람의 87%가 토큰 1칸보다 작습니다. 자동 검출 결과이므로 누락과 오검출이 섞여 있고, 사람이 검수한 정답이 아닙니다.

### 스크립트 (`masking/`)

| 파일 | 역할 |
|---|---|
| `gsam_window.py` | window 1개 마스킹과 시각화 (프롬프트·클래스 정의 포함) |
| `gsam_batch.py` | 전체 window 일괄 마스킹. 중단 후 다시 실행하면 이어서 진행 |
| `resume_mask.sh` | 서버 재부팅 후 자동 재시작용 |
| `make_exclude_noroad.py` | 도로 누락 이미지 제외 목록 생성 |
| `viz_noroad.py` | 도로 누락 이미지 표본 시각화 |
| `check_rope.py` | Alpamayo 프롬프트에서 이미지별 MRoPE 위치 ID 확인 |

스크립트 안의 경로는 실행한 서버 기준 그대로입니다. 다른 환경에서는 경로를 바꿔야 합니다.

```bash
python masking/gsam_batch.py --out ~/gil/data
```

## 2단계: Pruner 학습 (설계)

매 스텝 `이미지 → 고정 시각 인코더 → Pruner → 복원 디코더` 순서로 계산합니다.

| 항목 | 설정 | 출처 |
|---|---|---|
| Pruner 위치 | Spatial Merger 출력 이후, 언어 모델 입력 이전 | 직접 정함 |
| Query | 학습 가능한 벡터 1개 (D = 2048) | 논문 |
| PrunerLayer | 디코더 층 1개, full attention. Alpamayo LLM 0번 층으로 초기화 | 논문 구조, 초기화는 직접 정함 |
| 점수 | `S = Scorer(V* ⊙ Q*)`, Scorer는 선형층 1개 | 논문 |
| 학습 시 분할 | `S > 0` 기준, straight-through estimator | 논문 |
| 손실 | `0.2·(1−SSIM) + 0.8·MSE`, 전경·배경 가중 0.5씩 | 논문 |
| 학습 설정 | lr 2e-5, cosine, 10 epoch | 논문 |
| Pruner 범위 | window의 24장을 함께 봄 | 직접 정함 |
| 위치 정보 | Alpamayo LLM이 쓰는 MRoPE 위치 ID를 그대로 사용 | 직접 정함 |
| 복원 디코더 | 이미지별 복원, Transformer 6층, 토큰당 32×32 패치 출력 | 층 수만 논문, 나머지는 직접 정함 |
| 추론 | Top-K 선택, 위치 ID와 DeepStack 특징을 같은 인덱스로 선택 | 논문 + 직접 정함 |

함께 기록할 지표는 복원 품질, 선택 비율, 정답 전경과의 일치도, 객체 크기 구간별 토큰 보존율, 토큰 칸 안 전경 비율과 점수의 관계입니다.

## 환경

- Jetson Thor, Python 3.12, PyTorch 2.12, transformers 4.57
- 마스킹 모델 가중치는 Hugging Face에서 받습니다(약 3.5GB).

## 라이선스 관련

- Grounding DINO, SAM 2.1: Apache 2.0
- PhysicalAI-AV: NVIDIA Autonomous Vehicle Dataset License. 재배포와 파생물 공개가 제한되므로 이미지·마스크는 공개하지 않습니다.
