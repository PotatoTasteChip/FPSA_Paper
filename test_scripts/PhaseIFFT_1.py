import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class PhaseIFFT_1(nn.Module):
    """
    PhaseIFFT
    ----------
    • 입력 : RGB 텐서  [B, 3, H, W]  (0‒1 or 0‒255, float32/float16)
    • 처리 : 2-D FFT → 위상만 추출 → exp(jθ) → IFFT → 0‒1 정규화
    • 출력 : 입력과 동일한 shape·dtype  ([B, 3, H, W])  ← Conv 호환
    """

    def __init__(self, c1: int = 3, eps: float = 1e-6, norm: bool = True) -> None:
        """
        Parameters
        ----------
        c1   : 입력 채널 수 (기본 3)
        eps  : 정규화 시 0 나눔 방지용 값
        norm : True 면 0‒1 스케일로 정규화하여 반환
        """
        super().__init__()
        self.c1 = c1
        self.eps = eps
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:                 # x[B,C,H,W]
        # 1. 2-D FFT (오쏘노멀 정규화로 안정화)
        # 2-차원 푸리에 변환으로 공간 도메인 → 주파수 도메인. norm='ortho'는 직교 정규화를 적용해 에너지 보존(파서밸)이 쉬워집니다.
        F = fft.fft2(x, norm='ortho')                                   # complex64/128
        
        # 2. 위상만 추출
        # 복소수 F(u,v)=Ae^jθ에서 위상 θ만 가져옵니다. 위상은 형태·경계 정보를 담고 있어 구조적 특징을 보존합니다.
        phase = torch.angle(F)
                                                  # [-π, π]
        # 3. 진폭을 1 로 두고 복소수 재구성 → IFFT
        # 진폭을 1로 고정해서 e^(jθ)만 역변환. 위상-only 재구성은 윤곽·텍스처 위치는 유지하되 밝기(진폭) 정보는 제거합니다.
        x_rec = fft.ifft2(torch.exp(1j * phase), norm='ortho').real     # 실수부

        # 4. 선택적 0‒1 정규화 (배치·채널별)
        # 위상-only 이미지의 값 범위가 불규칙하므로 0‒1로 스케일링해 BatchNorm·Activation이 안정적으로 작동하도록 합니다.
        if self.norm:
            mn = x_rec.amin(dim=[2, 3], keepdim=True)
            mx = x_rec.amax(dim=[2, 3], keepdim=True)
            x_rec = (x_rec - mn) / (mx - mn + self.eps)

        # 원본이 FP16(AMP)라면 FP16, FP32라면 FP32로 맞춰 후속 레이어와 충돌이 없게 함.
        return x_rec.to(dtype=x.dtype)
          
# ── 1. 이미지 읽기 (OpenCV: BGR) ───────────────────────────────
img_bgr = cv2.imread("/home/lom-ljh/0_project/FPSA_Paper/FPSA_Paper/Dataset/4875/train/images/2025_02_27_144417_top_left_003150_jpg.rf.1a0496e0027355379ebef2b222fd2fcf.jpg")          # <— 테스트용 RGB 사진 경로
assert img_bgr is not None, "이미지 불러오기 실패!"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── 2. Torch Tensor로 변환 (0‒1 정규화) ────────────────────────
to_tensor = transforms.ToTensor()         # (H,W,C) → [C,H,W], float32 0‒1
x = to_tensor(img_rgb).unsqueeze(0)       # [1,3,H,W]  배치 차원 추가

# ── 3. PhaseIFFT 블록 실행 ───────────────────────────────────
phase_block = PhaseIFFT_1(norm=True).eval() # 학습 모드 X
with torch.no_grad():
    y = phase_block(x)                    # [1,3,H,W]

# ── 4. NumPy 로 변환해 시각화 ─────────────────────────────────
y_np = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")  # [H,W,C] 0‒255
y_bgr = cv2.cvtColor(y_np, cv2.COLOR_RGB2BGR)

cv2.imshow("Phase-only IFFT", y_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
