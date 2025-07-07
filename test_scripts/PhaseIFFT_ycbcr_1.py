import cv2, torch, torch.fft as fft
from torchvision import transforms
import torch.nn as nn

# ───────────────── Phase-only IFFT 모듈 (동일) ─────────────────
class PhaseIFFT(nn.Module):
    def __init__(self, eps=1e-6, norm=True):
        super().__init__()
        self.eps, self.norm = eps, norm
    def forward(self, x):                     # x:[B,C,H,W] (float,0-1)
        F   = fft.fft2(x, norm='ortho')       # 2-D FFT
        y   = fft.ifft2(torch.exp(1j*torch.angle(F)), norm='ortho').real
        if self.norm:                         # 0-1 정규화
            mn, mx = y.amin((-2,-1),True), y.amax((-2,-1),True)
            y = (y-mn) / (mx-mn+self.eps)
        return y.to(dtype=x.dtype)

# ───────────────── 1. 이미지 로딩 & YCbCr 변환 ────────────────
img_bgr = cv2.imread("/home/lom-ljh/0_project/FPSA_Paper/FPSA_Paper/Dataset/4875/train/images/2025_02_27_144417_top_left_003150_jpg.rf.1a0496e0027355379ebef2b222fd2fcf.jpg")                       # BGR
assert img_bgr is not None, "로드 실패!"
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb) # Y-Cr-Cb
Y, Cr, Cb = cv2.split(img_ycrcb)                       # 분리
img_ycbcr = cv2.merge([Y, Cb, Cr])                     # Y-Cb-Cr 순서

# ───────────────── 2. Tensor 변환 (0-1) ─────────────────────
x = transforms.ToTensor()(img_ycbcr).unsqueeze(0)      # [1,3,H,W]

# ───────────────── 3. Phase-only IFFT 실행 ─────────────────
phase_block = PhaseIFFT(norm=True).eval()
with torch.no_grad():
    y = phase_block(x)                                 # [1,3,H,W]

# ───────────────── 4. 시각화 ───────────────────────────────
out_np = (y.squeeze(0).permute(1,2,0).cpu().numpy()*255).astype("uint8")
out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)      # 다시 BGR
cv2.imshow("YCbCr-Phase-only", out_bgr)
cv2.waitKey(0); cv2.destroyAllWindows()
