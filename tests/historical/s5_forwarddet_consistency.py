"""S5: HoloForwardDet 数值一致性校验（协议门槛 3）

① forward 与 numpy 生成器确定性路径逐像素 MSE < 1e-4
② adjoint 伴随性 <D(x),y> ~= <x,D^T(y)>, atol = 1e-3
③ 尺寸：forward(HR) -> LR；adjoint(LR) -> HR
"""
import hashlib
import sys

print("SCRIPT_SHA256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())

import numpy as np
import torch
import cv2

sys.path.insert(0, "/root/autodl-tmp/holo_v7_baseline/restormer")
sys.path.insert(0, "/root/hao-d-wangluo/pipeline/scripts")

from basicsr.models.archs.holo_restormer_arch import (HoloForwardDet,
                                                     _lanczos_downsample)
import holo_degrade as hd


def mse(a, b):
    return float(((a - b) ** 2).mean())


hr = cv2.imread("/root/autodl-tmp/HR/263.bmp", cv2.IMREAD_COLOR)
hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
g = hr.mean(axis=2).astype(np.float32)          # 单通道，与生成器口径一致

SCALE = 4
SIG = 0.5
CUT_GEN = 1.0 / (2.0 * SCALE)                   # 生成器 cutoff = 0.125 cycles/px

print("HR 单通道 shape", g.shape, " 生成器 cutoff =", CUT_GEN)

# ---- 参考：生成器的确定性三步 ----
ref = hd.fft_lowpass(g, CUT_GEN, "circle")
ref_lo = ref.copy()
ref = hd.gaussian_blur(ref, SIG)
ref_lo2 = ref.copy()
ref = hd.downsample(ref, SCALE)
print("参考 LR shape", ref.shape)

x = torch.from_numpy(g)[None, None]

# ---- ① forward 对齐 ----
print()
print("== ① forward vs numpy 生成器 ==")
for cut in (1.0, 0.25):
    D = HoloForwardDet(scale=SCALE, cutoff=cut, aa_sigma=SIG)
    with torch.no_grad():
        out = D(x)[0, 0].numpy()
    ok = "PASS" if mse(out, ref) < 1e-4 else "FAIL"
    print(f"  HoloForwardDet(cutoff={cut:<4}) shape={out.shape} "
          f"MSE={mse(out, ref):.6e} max|d|={np.abs(out - ref).max():.6e}  [{ok}]")

# ---- 拆解：只比降采样算子 ----
print()
print("== 降采样算子单项对比（同样的低通输出，只换降采样）==")
a = hd.downsample(ref_lo2, SCALE)
with torch.no_grad():
    b = _lanczos_downsample(torch.from_numpy(ref_lo2)[None, None], SCALE)[0, 0].numpy()
print(f"  面积平均(cv2.INTER_AREA) vs Lanczos: MSE={mse(b, a):.6e} "
      f"max|d|={np.abs(b - a).max():.6e}")

# ---- ② 伴随性 ----
print()
print("== ② 伴随性 ==")
torch.manual_seed(0)
for cut in (0.25,):
    D = HoloForwardDet(scale=SCALE, cutoff=cut, aa_sigma=SIG)
    xh = torch.rand(1, 1, 256, 320, dtype=torch.float32)
    yl = torch.rand(1, 1, 64, 80, dtype=torch.float32)
    with torch.no_grad():
        dx = D.forward(xh)
        dtY = D.adjoint(yl)
        lhs = float((dx * yl).sum())
        rhs = float((xh * dtY).sum())
    d = abs(lhs - rhs)
    rel = d / max(abs(lhs), abs(rhs))
    ok = "PASS" if d < 1e-3 else "FAIL"
    print(f"  cutoff={cut}  forward {tuple(xh.shape)} -> {tuple(dx.shape)}"
          f"  adjoint {tuple(yl.shape)} -> {tuple(dtY.shape)}")
    print(f"  <D(x),y>={lhs:.6f}  <x,D^T(y)>={rhs:.6f}  |diff|={d:.3e} "
          f"rel={rel:.3e}  atol=1e-3 [{ok}]")
