# FatesGS 시스템 심층 분석

> **논문**: "FatesGS: Fast and Accurate Sparse-View Surface Reconstruction Using Gaussian Splatting with Depth-Feature Consistency"
> **학회**: AAAI 2025 Oral
> **저자**: Han Huang*, Yulun Wu* (공동 1저자), Chao Deng, Ge Gao (교신), Ming Gu, Yu-Shen Liu — Tsinghua University
> **arXiv**: 2501.04628

---

## 1. 연구 문제 정의

### 핵심 문제: Sparse-View 3D Surface Reconstruction

- **입력**: 매우 적은 수의 시점 이미지 (예: 3장)
- **출력**: 고품질 3D 메시(mesh) 표면 복원
- **난제**: 기존 3D/2D Gaussian Splatting은 dense view에서는 잘 작동하지만, sparse view에서는 Gaussian 프리미티브가 제한된 학습 뷰에 오버피팅되어 노이즈와 불완전한 표면을 생성함
- **기존 방법의 한계**:
  - NeuS, VolRecon 등 Neural implicit 방법: 장시간 per-scene optimization 또는 대규모 pre-training 필요
  - 3DGS/2DGS: sparse view에서 기하학적 정보 부족으로 표면 품질 급격히 저하

### FatesGS의 핵심 차별점

- pre-training 불필요 (학습된 prior 없이 작동)
- per-scene optimization이지만 매우 빠름 (기존 대비 60~200배 속도 향상)
- Gaussian Splatting 파이프라인의 장점을 최대한 활용하면서 sparse view 문제 해결

---

## 2. 전체 시스템 아키텍처

### 파이프라인 개요 (overview.png 기반)

```
Sparse Views (3장)
    │
    ├──→ COLMAP Initialization → Sparse Point Cloud → 2D Gaussians 초기화
    │
    ├──→ Monocular Depth Estimation (Marigold) → Depth Prior Maps
    │
    └──→ Multi-Level Feature Extraction (VisMVSNet FeatExt) → Feature Maps

    ↓ Training Loop (15,000 iterations)

    각 iteration:
    1. 랜덤 카메라 선택 → Differentiable Rasterization → Rendered Color + Depth + Normal
    2. Color Loss: L1 + SSIM (GT 이미지와 비교)
    3. Depth Ranking Loss (Lr): 모노큘러 깊이의 순위 관계 보존
    4. Depth Smoothness Loss (Ls): TV 기반 깊이 평활화
    5. Feature Consistency Loss (Lf): 멀티뷰 피처 일관성
    6. Normal + Distortion 정규화
    7. Adaptive Density Control (Densification & Pruning)

    ↓ After Training

    TSDF Fusion → Mesh Extraction → Post-processing
```

---

## 3. 핵심 기술 컴포넌트 상세 분석

### 3.1 기반 프레임워크: 2D Gaussian Splatting (2DGS)

FatesGS는 **2DGS** (diff-surfel-rasterization)를 기반으로 구축됨.

- **3DGS vs 2DGS**: 3DGS는 3D 타원체(ellipsoid) 사용, 2DGS는 2D 평면 디스크(surfel) 사용
- 2DGS가 표면 복원에 더 적합한 이유: surfel이 실제 표면에 더 잘 정렬됨
- **Gaussian 파라미터**: 위치(xyz), 색상(SH coefficients), 스케일(2D), 회전(quaternion), 불투명도(opacity)
- **렌더링 출력물**: RGB 이미지, 깊이맵(expected/median), 노멀맵, alpha맵, distortion맵

```python
# gaussian_renderer/__init__.py 핵심
# allmap에서 추출되는 정보:
render_alpha = allmap[1:2]          # 누적 불투명도
render_normal = allmap[2:5]         # 렌더링된 노멀
render_depth_median = allmap[5:6]   # 중간값 깊이
render_depth_expected = allmap[0:1] # 기대값 깊이
render_dist = allmap[6:7]           # 깊이 왜곡
```

### 3.2 Depth Ranking Loss (Lr) — Intra-view Depth Consistency

**목적**: 모노큘러 깊이 추정의 "순서(ranking)" 정보를 활용하여 렌더링된 깊이의 상대적 대소관계를 보존

**원리**:
- 모노큘러 깊이는 절대값이 부정확하지만, 픽셀 간 깊이의 상대적 순서는 신뢰할 수 있음
- 렌더링된 깊이에서 랜덤 픽셀 쌍을 샘플링하여, 모노큘러 깊이의 순위와 일치하도록 강제

**구현 상세** (`utils/loss_utils.py`):

```python
def patched_depth_ranking_loss(surf_depth, mono_depth, patch_size=32, margin=1e-4):
    # 1. 깊이맵을 32x32 패치로 분할
    surf_depth_patches = patchify(surf_depth, patch_size)  # [N, P*P]
    mono_depth_patches = patchify(mono_depth, patch_size)

    # 2. 패치 내 픽셀을 랜덤 셔플하여 쌍 생성
    rand_indices = torch.randperm(length)

    # 3. 순위 일관성 손실 계산
    # sign(mono_i - mono_j) * (surf_j - surf_i) + margin > 0 이면 손실 발생
    # → 모노큘러에서 i가 더 깊으면, 렌더링에서도 i가 더 깊어야 함
    patch_rank_loss = torch.max(
        sign(mono_A - mono_B) * (surf_B - surf_A) + margin,
        torch.zeros_like(...)
    ).mean()
```

**다중 크롭 전략**: 두 가지 크기의 CenterCrop (576x768, 544x736)을 적용하여 다양한 공간 범위에서 순위 일관성 확보

### 3.3 Feature Consistency Loss (Lf) — Multi-view Feature Consistency

**목적**: 렌더링된 깊이로부터 추정된 3D 표면 점들이 다른 뷰로 투영했을 때 피처가 일관되도록 강제

**이것이 FatesGS의 가장 핵심적인 기여**

**파이프라인**:

```
1. 렌더링된 깊이 → 3D 표면 점 추정 (depths_to_points)
2. 3D 점을 참조 뷰와 소스 뷰에 투영
3. 각 뷰의 피처맵에서 피처 샘플링 (grid_sample)
4. 코사인 유사도로 피처 일관성 측정
5. Visibility mask로 가려진 점 필터링
```

**Feature Extractor** (`utils/feat_utils.py: FeatExt`):
- **VisMVSNet**의 pre-trained feature extractor 사용 (frozen, 학습하지 않음)
- UNet 아키텍처: Conv(3→16) → UNet(16→32→64→128) → 3개 스케일 출력
- 가중치 파일: `utils/vismvsnet.pt` (MVS용 사전학습 가중치)
- 멀티스케일 피처 추출: scale 1 (1/4 해상도), scale 2 (1/8 해상도)

**피처 일관성 계산 상세**:

```python
def get_feat_loss_corr(diff_surf_pts, feat, cam, feat_src, src_cams, mask, scale):
    # 1. 3D 점을 각 카메라로 투영
    pts_img = idx_cam2img(idx_world2cam(pts_world, cam_pack), cam_pack)

    # 2. Visibility 마스크 생성
    #    - 각 소스 뷰에서 깊이 순서로 정렬
    #    - 같은 UV 좌표에 투영되는 점들 중 가장 가까운 것만 visible
    dists = torch.norm(diff_surf_pts - src_cam_centers, dim=-1)
    # unique UV에서 가장 가까운 점만 선택

    # 3. 피처 샘플링 (bilinear interpolation)
    gathered_feat = F.grid_sample(feat_pack, grid_n, mode='bilinear')

    # 4. 코사인 유사도 계산
    corr = (feat_ref * feat_src).sum() / (norm_ref * norm_src)
    corr_loss = (1 - corr).abs()

    # 5. outlier 필터링: corr_loss > 0.5인 점은 무시
    diff_mask = corr_loss < 0.5
```

**Pair 관계**: `pair.txt`에서 각 이미지의 상위 2개 소스 뷰 정보 로드 → DTU 데이터셋 특화

### 3.4 Depth Smoothness Loss (Ls)

**목적**: 모노큘러 깊이의 에지 정보를 활용한 edge-aware depth smoothing

```python
def TVLoss(network_output, pred_output, edge_margin=1e-2, margin=1e-4):
    # 수평/수직 방향 깊이 차이 계산
    # 단, 모노큘러 깊이에서 에지가 있는 곳(차이 > edge_margin)은 스무딩하지 않음
    h_diff = max(|surf_h_diff| - margin, 0) * (|mono_h_diff| < edge_margin)
    w_diff = max(|surf_w_diff| - margin, 0) * (|mono_w_diff| < edge_margin)
```

### 3.5 기타 정규화

- **Normal Consistency Loss** (iteration > 7000): 렌더링된 노멀과 깊이 기반 표면 노멀의 일관성
- **Distortion Loss** (iteration > 3000): Gaussian ray상의 분포 집중도 → 표면에 가까운 렌더링 유도

---

## 4. 전체 손실 함수

```python
total_loss = L_color                          # L1 + λ_dssim * (1 - SSIM)
           + dist_loss                         # λ_dist * mean(rend_dist), iter > 3000
           + normal_loss                       # λ_normal * (1 - dot(rend_normal, surf_normal)), iter > 7000
           + dsmooth_loss                      # TVLoss (edge-aware smoothing)
           + λ_feat * feat_loss                # Multi-view feature consistency
           + λ_depth * depth_rank_loss         # Depth ranking consistency
```

### 하이퍼파라미터 (arguments/__init__.py)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `iterations` | 15,000 | 총 학습 반복 횟수 |
| `lambda_dssim` | 0.2 | SSIM 가중치 |
| `lambda_dist` | 10,000 | Distortion 정규화 가중치 |
| `lambda_normal` | 0.05 | Normal 일관성 가중치 |
| `lambda_feat` | 1.5 | **Feature consistency 가중치** |
| `lambda_depth` | 10.0 | **Depth ranking 가중치** |
| `densify_until_iter` | 15,000 | Densification 끝 iteration |
| `densification_interval` | 100 | Densification 주기 |
| `opacity_cull` | 0.1 | Opacity pruning 임계값 |

---

## 5. 데이터 구조 및 전처리

### 5.1 DTU 데이터셋 구조

```
DTU/
└── set_23_24_33/           # 뷰 세트 (학습에 사용할 3개 뷰의 ID)
    └── scan24/             # 스캔 장면
        ├── pair.txt        # 각 이미지별 소스 뷰 쌍 정보
        ├── images/         # 원본 이미지 (1600x1200)
        ├── sparse/0/       # COLMAP sparse reconstruction
        │   ├── cameras.txt
        │   ├── images.txt
        │   └── points3D.txt
        ├── dense/          # COLMAP dense reconstruction
        │   └── fused.ply   # 초기 포인트 클라우드 (학습 시 사용)
        └── depth_npy/      # Marigold 모노큘러 깊이 추정 결과
            ├── 0000_pred.npy
            └── ...
```

### 5.2 전처리 요구사항

1. **COLMAP**: sparse/dense reconstruction 완료 필요
2. **Marigold**: 모노큘러 깊이 추정 → `.npy` 형식 저장
3. **pair.txt**: MVS 방식의 뷰 페어링 정보 (각 이미지당 n개의 소스 뷰와 매칭 스코어)
4. **해상도**: `-r 2` 옵션으로 1/2 해상도 (800x600)에서 학습

### 5.3 학습 뷰 선택 (dataset_readers.py)

```python
# DTU 표준 sparse-view 프로토콜
train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]  # 후보 뷰
train_idx = train_idx[:n_views]  # 기본 3개 뷰 사용 (25, 22, 28)
exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]  # 제외 뷰
test_idx = [나머지]  # 평가용
```

---

## 6. 실험 파이프라인

### 6.1 학습 (train.py + train.sh)

```bash
# 단일 스캔 학습
SCAN="24"
EXP_NAME="fatesgs_exp"
CUDA_VISIBLE_DEVICES=0 python train.py \
    -s data/DTU/set_23_24_33/scan${SCAN} \
    -m output/DTU/set_23_24_33/${EXP_NAME}/scan${SCAN} \
    -r 2  # 1/2 해상도
```

**학습 과정**:
1. COLMAP dense fused.ply에서 초기 Gaussian 생성
2. 15,000 iteration 학습 (3개 뷰 반복 샘플링)
3. 100 iter마다 densification, 1000 iter마다 opacity reset
4. 7,000 / 15,000 iter에서 체크포인트 저장

### 6.2 메시 추출 (render.py)

```bash
CUDA_VISIBLE_DEVICES=0 python render.py \
    -s data/DTU/set_23_24_33/scan24 \
    -m output/DTU/set_23_24_33/fatesgs_exp/scan24 \
    -r 2 --skip_test --skip_train
```

**메시 추출 과정**:
1. 학습된 Gaussian 로드
2. 학습 카메라에서 깊이맵 렌더링
3. **TSDF (Truncated Signed Distance Function) Fusion**: Open3D의 ScalableTSDFVolume 사용
4. Marching Cubes로 메시 추출
5. Post-processing: 작은 연결 컴포넌트 제거 (상위 50개 클러스터 유지)

### 6.3 메시 평가 (eval/eval_dtu/dtu_eval.py)

```bash
python eval/eval_dtu/dtu_eval.py \
    --input_mesh output/.../fuse_post.ply \
    --output_dir output/.../ \
    --mask <mask_dir> \
    --DTU <gt_dir>
```

**평가 과정**:
1. **Mesh Culling**: 마스크 기반으로 메시에서 배경 정점 제거 (binary dilation with disk(24))
2. **Chamfer Distance 계산**:
   - d2s (data-to-scan): 추출된 메시 → GT 포인트 클라우드 거리
   - s2d (scan-to-data): GT → 추출된 메시 거리
   - Overall = (d2s + s2d) / 2
3. **15개 표준 DTU 스캔**: [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

### 6.4 렌더링 품질 평가 (metrics_dtu.py)

- PSNR, SSIM, LPIPS (VGG) 측정
- 마스크 적용된 영역에서만 평가

---

## 7. 코드 구조 요약

```
FatesGS/
├── train.py                 # 메인 학습 루프
├── render.py                # 메시 추출 + 렌더링
├── convert.py               # COLMAP 전처리 도구
├── metrics_dtu.py           # 렌더링 품질 메트릭 (PSNR/SSIM/LPIPS)
├── train.sh / eval.sh       # 실행 스크립트
│
├── arguments/
│   └── __init__.py          # ModelParams, OptimizationParams, PipelineParams
│
├── scene/
│   ├── __init__.py          # Scene 클래스 (데이터 로딩 + 카메라 관리)
│   ├── gaussian_model.py    # GaussianModel (파라미터 + densification)
│   ├── dataset_readers.py   # COLMAP/Blender 데이터 로딩 + 피처 추출
│   ├── cameras.py           # Camera 클래스 (내/외부 파라미터 + 피처 저장)
│   └── colmap_loader.py     # COLMAP 형식 파서
│
├── gaussian_renderer/
│   └── __init__.py          # render() 함수 (2DGS 래스터라이저 호출)
│
├── utils/
│   ├── feat_utils.py        # ★ FeatExt (VisMVSNet 피처) + Feature Loss 계산
│   ├── loss_utils.py        # ★ Depth Ranking Loss + TVLoss + SSIM
│   ├── point_utils.py       # depths_to_points, depth_to_normal
│   ├── mesh_utils.py        # GaussianExtractor (TSDF fusion + 메시 추출)
│   ├── camera_utils.py      # 카메라 로딩 유틸리티
│   ├── graphics_utils.py    # 변환 행렬 유틸리티
│   ├── render_utils.py      # 비디오 생성 + 경로 생성
│   ├── general_utils.py     # 일반 유틸리티
│   ├── sh_utils.py          # Spherical Harmonics 유틸리티
│   ├── mcube_utils.py       # Marching Cubes (unbounded 메시용)
│   └── vismvsnet.pt         # VisMVSNet pre-trained 가중치 (frozen)
│
├── submodules/
│   ├── diff-surfel-rasterization/  # 2DGS CUDA 래스터라이저
│   └── simple-knn/                  # KNN CUDA 연산
│
├── eval/
│   └── eval_dtu/            # DTU 메시 평가 스크립트
│
└── lpipsPyTorch/            # LPIPS 메트릭 구현
```

---

## 8. 실험 재현을 위한 실행 가이드

### 8.1 환경 설정

```bash
conda create -n fatesgs python=3.8
conda activate fatesgs

# PyTorch (CUDA 필수)
pip install torch==2.0.0 torchvision==0.15.1

# 기타 의존성
pip install -r requirements.txt

# CUDA 서브모듈 빌드 (자동)
# diff-surfel-rasterization, simple-knn은 pip install 시 CUDA 컴파일
# pytorch3d도 소스에서 빌드
```

### 8.2 데이터 준비

```bash
# 1. DTU 데이터셋 다운로드 (Google Drive 링크)
# 2. Marigold로 모노큘러 깊이 추정
cd Marigold
python run.py --input_dir <DTU_images> --output_dir <DTU_depth_npy>
# 출력: XXXX_pred.npy 파일들을 depth_npy/ 폴더에 배치

# 3. 깊이맵 해상도 확인
# 학습 시 -r 2 사용 → 깊이맵도 800x600이어야 함
```

### 8.3 전체 실험 파이프라인

```bash
# DTU 15개 스캔 전체 실험
for SCAN in 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122; do
    # 학습 + 메시 추출
    bash train.sh $SCAN fatesgs_v1 0
done

# 메시 품질 평가 (Chamfer Distance)
bash eval.sh 0 fatesgs_v1

# 렌더링 품질 평가 (PSNR/SSIM/LPIPS)
python metrics_dtu.py -m output/DTU/set_23_24_33/fatesgs_v1/scan24 [...]
```

### 8.4 주요 실험 변수 및 조정 포인트

| 실험 변수 | 위치 | 설명 |
|-----------|------|------|
| `lambda_feat` | arguments | Feature loss 강도 (기본 1.5) |
| `lambda_depth` | arguments | Depth ranking loss 강도 (기본 10.0) |
| `n_views` | dataset_readers.py:226 | 학습 뷰 수 (기본 3) |
| `pair[:2]` | dataset_readers.py:174 | 소스 뷰 수 (기본 2) |
| Depth estimator | 외부 | Marigold vs 더 고급 모델 |
| `mesh_res` | render.py | 메시 해상도 (기본 1024) |
| `num_cluster` | render.py | 후처리 시 유지할 클러스터 수 (기본 50) |
| `iterations` | arguments | 학습 횟수 (기본 15,000) |
| 해상도 (`-r`) | CLI | 이미지 다운스케일 (기본 2 = 절반) |

---

## 9. 핵심 인사이트 및 기술적 포인트

### 9.1 왜 Feature Consistency가 효과적인가?

- **모노큘러 깊이**: 상대적 순서만 정확, 절대 깊이는 부정확 → ranking loss만으로는 부족
- **Feature consistency**: 3D 점의 "절대 위치"를 간접적으로 제약
  - 깊이가 정확해야 → 3D 점이 올바른 위치 → 다른 뷰에서 같은 피처
  - 깊이가 부정확하면 → 3D 점 위치 이탈 → 피처 불일치 → 큰 loss
- **VisMVSNet 피처 사용 이유**: MVS 태스크용으로 사전학습되어 기하학적 정보를 잘 인코딩

### 9.2 Visibility 처리의 중요성

```python
# feat_utils.py에서의 visibility mask 생성
# 같은 UV에 투영되는 여러 3D 점 중 가장 가까운 것만 visible로 처리
# → self-occlusion으로 인한 잘못된 피처 비교 방지
```

### 9.3 2DGS 선택의 이유

- 2D surfel은 표면에 정렬되어 메시 추출에 유리
- scaling이 2D (두 축만) → 3D 대비 파라미터 효율적
- TSDF fusion과의 궁합이 좋음

### 9.4 Dense Point Cloud 초기화

- COLMAP의 sparse reconstruction이 아닌 **dense reconstruction (fused.ply)**로 초기화
- Sparse view에서도 초기 기하학 정보가 더 풍부

---

## 10. 예상되는 실험 결과 및 벤치마크

### DTU 데이터셋 (Chamfer Distance, mm)

논문 기준 FatesGS는 기존 sparse-view 방법 대비:
- **속도**: SparseNeuS, VolRecon 등 대비 60~200x 빠름
- **품질**: Chamfer Distance 경쟁력 있거나 우수
- **학습 시간**: ~수 분 (GPU 1장, 15K iterations)

### 비교 대상 방법들

- **Neural Implicit**: NeuS, VolRecon, SparseNeuS, GenS
- **Gaussian Splatting**: 3DGS, 2DGS, DNGaussian, PGSR
- **Feed-forward**: MVSNet 계열

---

## 11. 한계점 및 개선 가능 방향

1. **DTU 특화**: pair.txt, 뷰 인덱스 등이 DTU에 하드코딩된 부분 있음
2. **Marigold 의존**: 깊이 추정 품질이 최종 결과에 직접 영향
3. **VisMVSNet 피처 고정**: fine-tuning 없이 frozen → 도메인 갭 가능성
4. **Bounded scene 가정**: TSDF fusion이 bounded scene에 최적화
5. **3-view 고정**: 뷰 수에 따른 적응적 전략 부재

---

## 12. 빠른 실험 체크리스트

- [ ] CUDA 환경 확인 (CUDA 11.x + PyTorch 2.0)
- [ ] conda 환경 생성 및 의존성 설치
- [ ] 서브모듈 CUDA 빌드 확인 (diff-surfel-rasterization, simple-knn)
- [ ] pytorch3d 빌드 확인
- [ ] DTU 데이터셋 다운로드 및 구조 확인
- [ ] Marigold 깊이 추정 실행 → depth_npy/ 배치
- [ ] 깊이맵 해상도 확인 (학습 해상도와 일치)
- [ ] vismvsnet.pt 가중치 파일 존재 확인
- [ ] pair.txt 존재 확인
- [ ] 단일 스캔(scan24)으로 학습 테스트
- [ ] 메시 추출 및 시각화
- [ ] 전체 15개 스캔 평가 실행
