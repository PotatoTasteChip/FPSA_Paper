import cv2
import torch
import torch.fft as fft
import torch.nn as nn
from torchvision import transforms

class PhaseIFFT(nn.Module):
    def __init__(self, eps=1e-6, norm=True):
        super().__init__()
        self.eps = eps
        self.norm = norm

    def forward(self, x):                     # x: [B,1,H,W]
        F = fft.fft2(x, norm='ortho')         # 2D FFT
        phase = torch.angle(F)                # 위상만 추출
        x_rec = fft.ifft2(torch.exp(1j * phase), norm='ortho').real  # 진폭=1로 IFFT

        if self.norm:
            mn = x_rec.amin(dim=[2, 3], keepdim=True)
            mx = x_rec.amax(dim=[2, 3], keepdim=True)
            x_rec = (x_rec - mn) / (mx - mn + self.eps)

        return x_rec.to(dtype=x.dtype)

# ── 1. 이미지 로딩 & Y 채널 추출 ───────────────────────────────
img_bgr = cv2.imread("/home/lom-ljh/0_project/FPSA_Paper/FPSA_Paper/Dataset/4875/train/images/2025_02_27_144417_top_left_003150_jpg.rf.1a0496e0027355379ebef2b222fd2fcf.jpg")  # ← 경로 수정
assert img_bgr is not None, "이미지 불러오기 실패!"
img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
y_channel = img_ycrcb[:, :, 0]  # Y 채널만 추출

# ── 2. Torch Tensor로 변환 ────────────────────────────────────
to_tensor = transforms.ToTensor()  # 0–1, [H,W] → [1,H,W]
x = to_tensor(y_channel).unsqueeze(0)  # → [1,1,H,W]

# ── 3. Phase-only IFFT 처리 ──────────────────────────────────
block = PhaseIFFT(norm=True).eval()
with torch.no_grad():
    y = block(x)  # [1,1,H,W]

# ── 4. NumPy로 변환 후 시각화 ────────────────────────────────
y_np = (y.squeeze(0).squeeze(0).cpu().numpy() * 255).astype("uint8")  # [H,W]
cv2.imshow("Phase-only IFFT (Y channel)", y_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
