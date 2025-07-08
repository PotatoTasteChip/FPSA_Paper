import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
# 0. Letterbox Padding 함수 (PIL → PIL)
def letterbox_pad(img, out_size=640, fill=0):
    w, h = img.size
    scale = out_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (out_size, out_size), (fill, fill, fill))
    paste_x = (out_size - new_w) // 2
    paste_y = (out_size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    return canvas

# ────────────────────────────────────────────────
# 1. PhaseIFFT_1 클래스
class PhaseIFFT_1(nn.Module):
    def __init__(self, c1, c2=1, keep_rgb_channels=False,
                 eps=1e-6, norm=True,
                 cut_low=0.1, cut_high=0.4,
                 return_mode='stack'):
        super().__init__()
        self.c2, self.keep_rgb = c2, keep_rgb_channels
        self.eps, self.norm = eps, norm
        self.cut_low, self.cut_high = cut_low, cut_high
        self.return_mode = return_mode
        self.register_buffer("rgb2y",
                             torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1))

    def _make_masks(self, H, W, device):
        fy = torch.fft.fftfreq(H, device=device).view(-1, 1).repeat(1, W)
        fx = torch.fft.fftfreq(W, device=device).view(1, -1).repeat(H, 1)
        r = (fx ** 2 + fy ** 2).sqrt()
        r = torch.fft.fftshift(r)
        low_mask  = (r <= self.cut_low)
        mid_mask  = (r >  self.cut_low) & (r <= self.cut_high)
        high_mask = (r >  self.cut_high)
        return low_mask, mid_mask, high_mask

    def _phase_ifft(self, F, mask):
        mask = torch.fft.ifftshift(mask)
        #comp = torch.exp(1j * torch.angle(F)) * mask
        comp = F * mask
        y = torch.fft.ifft2(comp, norm='ortho').real
        if self.norm:
            mn = y.amin((2, 3), keepdim=True)
            mx = y.amax((2, 3), keepdim=True)
            y = (y - mn) / (mx - mn + self.eps)
        return y

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, band=None):
        dtype_in = x.dtype
        if x.shape[1] == 3:
            x = (x * self.rgb2y.to(x.dtype)).sum(1, keepdim=True)
        F = torch.fft.fft2(x.float(), norm='ortho')
        low_m, mid_m, high_m = self._make_masks(x.shape[2], x.shape[3], x.device)

        if self.return_mode == 'stack':
            outs = [self._phase_ifft(F, low_m),
                    self._phase_ifft(F, mid_m),
                    self._phase_ifft(F, high_m)]
            y = torch.cat(outs, dim=1)
        else:
            if band is None:
                raise ValueError("band 인자를 지정하세요.")
            m = {'low': low_m, 'mid': mid_m, 'high': high_m}[band]
            y = self._phase_ifft(F, m)

        if self.c2 > y.shape[1]:
            y = y.repeat(1, self.c2 // y.shape[1], 1, 1)
        return y.to(dtype_in)

# ────────────────────────────────────────────────
# 2. 이미지 불러오기 및 640×640 패딩
img_path = "/home/lom-ljh/0_project/FPSA_Paper/FPSA_Paper/Dataset/test_syj_951/test/images/2025_06_13_152259_top_right_000481_jpg.rf.d8e7eb1b4f1c6b9507fe7165bf7d5dd5.jpg"  # ← 실제 이미지 경로로 바꿔주세요
img_raw = Image.open(img_path).convert("RGB")
img = letterbox_pad(img_raw, out_size=640)

# 3. Tensor 변환 [1,3,640,640], 값 0~1
to_tensor = T.ToTensor()
x = to_tensor(img).unsqueeze(0)

# ────────────────────────────────────────────────
# 4. 주파수 스펙트럼 시각화 함수
def vis_spectrum(F):
    mag = torch.abs(torch.fft.fftshift(F, dim=(-2, -1)))
    mag = torch.log1p(mag)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag

# ────────────────────────────────────────────────
# 5. PhaseIFFT 실행 및 마스크 확인
phase_layer = PhaseIFFT_1(c1=3, c2=1, return_mode='stack',
                          cut_low=0.1, cut_high=0.4)

with torch.no_grad():
    x_y = (x * phase_layer.rgb2y).sum(1, keepdim=True)
    F = torch.fft.fft2(x_y.float(), norm='ortho')

    low_m, mid_m, high_m = phase_layer._make_masks(x.shape[2], x.shape[3], x.device)

    spec_orig  = vis_spectrum(F)[0, 0]
    spec_low   = vis_spectrum(F * torch.fft.ifftshift(low_m))[0, 0]
    spec_mid   = vis_spectrum(F * torch.fft.ifftshift(mid_m))[0, 0]
    spec_high  = vis_spectrum(F * torch.fft.ifftshift(high_m))[0, 0]

    y = phase_layer(x)  # [1,3,640,640]

# ────────────────────────────────────────────────
# 6. 시각화
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

# 주파수 도메인
specs = [spec_orig, spec_low, spec_mid, spec_high]
titles = ["|F| (full)", "Low mask", "Mid mask", "High mask"]
for i in range(4):
    axs[0, i].imshow(specs[i].cpu(), cmap="gray")
    axs[0, i].set_title(titles[i])
    axs[0, i].axis("off")

# 공간 도메인
axs[1, 0].imshow(img)
axs[1, 0].set_title("Original")
axs[1, 0].axis("off")
for i in range(3):
    axs[1, i+1].imshow(y[0, i].cpu(), cmap="gray")
    axs[1, i+1].set_title(["Low-freq", "Mid-freq", "High-freq"][i])
    axs[1, i+1].axis("off")

plt.tight_layout()
plt.show()
