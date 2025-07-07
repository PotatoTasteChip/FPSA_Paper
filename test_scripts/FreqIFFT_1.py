import cv2
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class FreqIFFT_1(nn.Module):
    """
    진폭(주파수 크기)만 사용하여 재구성하는 IFFT 블록

    입력: [B, C, H, W]  (RGB 이미지)
    처리: 2D FFT → 복소수 크기만 추출 → IFFT → 정규화
    출력: 원래와 동일한 shape [B, C, H, W]
    """

    def __init__(self, c1: int = 3, eps: float = 1e-6, norm: bool = True) -> None:
        super().__init__()
        self.c1 = c1
        self.eps = eps
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ① FFT
        F = fft.fft2(x, norm="ortho")                         # complex
        # ② 진폭(A=|F|)만 추출, 위상 제거
        amplitude = torch.abs(F)
        # ③ IFFT (위상 제거 → 실수 입력으로 복소수 아님)
        x_rec = fft.ifft2(amplitude, norm="ortho").real
        # ④ 정규화 (선택)
        if self.norm:
            mn = x_rec.amin(dim=[2, 3], keepdim=True)
            mx = x_rec.amax(dim=[2, 3], keepdim=True)
            x_rec = (x_rec - mn) / (mx - mn + self.eps)
        return x_rec.to(dtype=x.dtype)

          
# ── 1. 이미지 읽기 (OpenCV: BGR) ───────────────────────────────
img_bgr = cv2.imread("/home/lom-ljh/0_project/FPSA_Paper/FPSA_Paper/Dataset/4875/train/images/2025_02_27_144417_top_left_003150_jpg.rf.1a0496e0027355379ebef2b222fd2fcf.jpg")          # <— 테스트용 RGB 사진 경로
assert img_bgr is not None, "이미지 로드 실패!"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ── 2. Torch Tensor로 변환: [1, 3, H, W]
to_tensor = transforms.ToTensor()
x = to_tensor(img_rgb).unsqueeze(0)  # [B,C,H,W] = [1,3,H,W]

# ── 3. FreqIFFT_1 블록 실행
block = FreqIFFT_1(norm=True).eval()
with torch.no_grad():
    y = block(x)  # [1,3,H,W]
    y_min = y.amin(dim=[2, 3], keepdim=True)  # 최소값 [B, C, 1, 1]
    y_max = y.amax(dim=[2, 3], keepdim=True)  # 최대값 [B, C, 1, 1]

    y = (y - y_min) / (y_max - y_min + 1e-6)  # 안정적인 정규화

# ── 4. 결과 시각화를 위해 NumPy로 변환
y_np = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")  # [H, W, C]
y_bgr = cv2.cvtColor(y_np, cv2.COLOR_RGB2BGR)

# ── 5. 시각화
cv2.imshow("Freq-only IFFT", y_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
