#!/usr/bin/env python3
"""
生成统一实验配置：所有模型共用同一套数据、同一套训练超参，只有网络结构不同。

这是"所有实验都用的实验脚本"的落点 —— 你只改 --models 列表，其余完全一致。

用法
  python gen_configs.py --data_root data/holo --out configs --scale 4

产出
  configs/train/<MODEL>_<protocol>_x<S>.yml
  configs/test/<MODEL>_<protocol>_x<S>.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from exp_config import load as load_exp_cfg, find_config, paths_abs
    _HAS_EXP_CFG = True
except Exception:
    _HAS_EXP_CFG = False

# ---------------------------------------------------------------------------
# 模型注册表
#   type     : 注册在 basicsr/archs 里的类名（需要先把对应 arch 文件放进去）
#   arch     : 网络结构参数（1 通道灰度，X4）
#   source   : arch 文件来源
#   note     : 备注
# ---------------------------------------------------------------------------
MODELS = {
    # ---- 零接入：上游 BasicSR 自带，pip install basicsr 后直接可用 ----
    'SRResNet': dict(
        type='MSRResNet',
        arch=dict(num_in_ch=1, num_out_ch=1, num_feat=64, upscale=4),
        source='上游 BasicSR 自带 (basicsr/archs/srresnet_arch.py)',
        note='最快出结果，~1.5M，用来先跑通全流程'),

    'EDSR': dict(
        type='EDSR',
        arch=dict(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=16,
                  upscale=4, res_scale=1.0),
        source='上游 BasicSR 自带 (basicsr/archs/edsr_arch.py)',
        note='经典 CNN 基准，快'),

    'RCAN': dict(
        type='RCAN',
        arch=dict(num_in_ch=1, num_out_ch=1, num_feat=64, num_group=10,
                  num_block=20, squeeze_factor=16, upscale=4),
        source='上游 BasicSR 自带 (basicsr/archs/rcan_arch.py)',
        note='★ 域内基准研究用的就是 RCAN，必比'),

    'RRDBNet': dict(
        type='RRDBNet',
        arch=dict(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=23,
                  num_grow_ch=32, scale=4),
        source='上游 BasicSR 自带 (basicsr/archs/rrdbnet_arch.py)',
        note='ESRGAN 主干，PSNR 基线常用'),

    'SwinIR': dict(
        type='SwinIR',
        arch=dict(upscale=4, img_size=64, in_chans=1, img_range=1.,
                  depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                  num_heads=[6, 6, 6, 6, 6, 6], window_size=8, mlp_ratio=2,
                  upsampler='pixelshuffle', resi_connection='1conv'),
        source='上游 BasicSR 自带 (basicsr/archs/swinir_arch.py)',
        note='★ 域内基准 + H2HSR 用的就是它，必比'),

    # ---- 需接入：把单文件放进 basicsr/archs/ 并 import，约 5 分钟 ----
    'HAT': dict(
        type='HAT',
        arch=dict(upscale=4, img_size=64, in_chans=1, img_range=1.,
                  embed_dim=180, depths=[6, 6, 6, 6, 6, 6],
                  num_heads=[6, 6, 6, 6, 6, 6], window_size=16, mlp_ratio=2,
                  upsampler='pixelshuffle', resi_connection='1conv'),
        source='github.com/XPixelGroup/HAT -> basicsr/archs/hat_arch.py',
        note='通用 Transformer SOTA，但重（~20M），训练慢'),

    'NAFNet': dict(
        type='NAFNet',
        arch=dict(img_channel=1, width=32, middle_blk_num=12,
                  enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
                  upscale=4),
        source='github.com/megvii-research/NAFNet -> basicsr/models/archs/NAFNet_arch.py',
        note='轻量高效，主张效率时的对手'),

    'DAT': dict(
        type='DAT',
        arch=dict(upscale=4, in_chans=1, img_size=64, img_range=1.,
                  split_size=[8, 32], depth=[6, 6, 6, 6, 6, 6], embed_dim=180,
                  num_heads=[6, 6, 6, 6, 6, 6], expansion_factor=4,
                  resi_connection='1conv'),
        source='github.com/zhengchen1999/DAT（本仓库自带）',
        note='全息领域真空，放进消融表当"直接搬通用 SOTA"的对照，不做主基线'),

    # ---- 我们的方法：从 Restormer 改造而来 ----
    'HoloRestormer': dict(
        type='HoloRestormer',
        arch=dict(upscale=4, img_size=64, in_chans=1, img_range=1.,
                  dim=48, num_blocks=[4, 6, 6, 8],
                  num_refinement_blocks=4, heads=[1, 2, 4, 8],
                  ffn_expansion_factor=2.66, bias=False,
                  LayerNorm_type='BiasFree', n_bands=6),
        source='自研：Restormer 的 MDTA 通道轴 -> 频带轴（见桌面《全息超分-基线选型与改造设计.md》第 4 节）',
        note='★ 我们的方法：频带注意力 + 方向性卷积 + 物理一致性损失'),

    # ---- 需装 mamba-ssm（要编译，装环境最慢）----
    'MambaIR': dict(
        type='MambaIR',
        arch=dict(upscale=4, img_size=64, in_chans=1, img_range=1.,
                  embed_dim=180, d_state=16, depths=[6, 6, 6, 6, 6, 6],
                  img_size_2=64),
        source='github.com/csguoh/MambaIR（需 pip install mamba-ssm causal-conv1d）',
        note='⭐ HoloMamba 证明 SSM 在全息有优势，这条对照最有说服力，但不快'),
}

# ---------------------------------------------------------------------------
# 统一训练超参：从「实验总配置」读，保证所有实验完全一致
# 找不到总配置文件时才回落到这里的默认值
# ---------------------------------------------------------------------------
def build_common(cfg: dict | None = None) -> dict:
    d = dict(
        manual_seed=10, gt_size=256, batch_size_per_gpu=8,
        num_worker_per_gpu=8, total_iter=250000, lr=1e-4,
        betas='[0.9, 0.99]', milestones='[125000, 200000, 225000, 237500]',
        val_freq=5000, save_checkpoint_freq=5000,
        use_hflip='true', use_rot='true', loss_type='L1Loss',
    )
    if not cfg:
        return d
    tr = cfg['training']
    op = tr['optimizer']
    sc = tr['scheduler']
    aug = tr.get('augmentation', {})
    d.update(
        manual_seed=tr['seeds'][0],
        gt_size=tr['gt_size'],
        batch_size_per_gpu=tr['batch_size_per_gpu'],
        num_worker_per_gpu=tr['num_worker_per_gpu'],
        total_iter=tr['total_iter'],
        lr=op['lr'],
        betas='[' + ', '.join(str(b) for b in op['betas']) + ']',
        milestones='[' + ', '.join(str(m) for m in sc['milestones']) + ']',
        gamma=sc['gamma'],
        val_freq=tr['val_freq'],
        save_checkpoint_freq=tr['save_checkpoint_freq'],
        use_hflip=str(aug.get('use_hflip', True)).lower(),
        use_rot=str(aug.get('use_rot', True)).lower(),
        loss_type=tr['loss']['type'],
    )
    return d


# 回落到默认值时的常量（无总配置文件）
COMMON = build_common(None)

TRAIN_TPL = """\
# 由 gen_configs.py 自动生成 —— 不要手改，改脚本后重新生成
# 模型: {model}   协议: {protocol}   倍数: X{scale}
# {note}
# arch 来源: {source}
name: train_{model}_{protocol}_x{scale}
model_type: {model_type}
scale: {scale}
num_gpu: auto
manual_seed: {manual_seed}

datasets:
  train:
    task: SR
    name: holo_{protocol}_x{scale}
    type: PairedImageDataset
    dataroot_gt: {data_root}/HR/train
    dataroot_lq: {data_root}/LR_{protocol}/X{scale}/train
    filename_tmpl: '{{}}'
    io_backend:
      type: disk
    gt_size: {gt_size}
    use_hflip: {use_hflip}
    use_rot: {use_rot}
    use_shuffle: true
    num_worker_per_gpu: {num_worker_per_gpu}
    batch_size_per_gpu: {batch_size_per_gpu}
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    task: SR
    name: holo_{protocol}_x{scale}_val
    type: PairedImageDataset
    dataroot_gt: {data_root}/HR/val
    dataroot_lq: {data_root}/LR_{protocol}/X{scale}/val
    filename_tmpl: '{{}}'
    io_backend:
      type: disk

network_g:
{arch_block}

path:
  pretrain_network_g: ~
  strict_load_g: false
  resume_state: ~

train:
  optim_g:
    type: Adam
    lr: !!float {lr}
    weight_decay: 0
    betas: {betas}
  scheduler:
    type: MultiStepLR
    milestones: {milestones}
    gamma: {gamma}
  total_iter: {total_iter}
  warmup_iter: -1
  pixel_opt:
    type: {loss_type}
    loss_weight: 1.0
    reduction: mean

val:
  val_freq: !!float {val_freq}
  save_img: false
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: {scale}
      test_y_channel: false      # 灰度强度图，不做 RGB->Y 转换
    ssim:
      type: calculate_ssim
      crop_border: {scale}
      test_y_channel: false

logger:
  print_freq: 200
  save_checkpoint_freq: !!float {save_checkpoint_freq}
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

dist_params:
  backend: nccl
  port: 29500
"""

TEST_TPL = """\
# 由 gen_configs.py 自动生成
name: test_{model}_{protocol}_x{scale}
model_type: {model_type}
scale: {scale}
num_gpu: 1
manual_seed: 10

datasets:
  test_1:
    task: SR
    name: holo_{protocol}_x{scale}_test
    type: PairedImageDataset
    dataroot_gt: {data_root}/HR/test
    dataroot_lq: {data_root}/LR_{protocol}/X{scale}/test
    filename_tmpl: '{{}}'
    io_backend:
      type: disk

network_g:
{arch_block}

path:
  pretrain_network_g: experiments/{model}_{protocol}_x{scale}/models/net_g_{total_iter}.pth
  strict_load_g: true
  param_key: params

val:
  save_img: true
  suffix: ~
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: {scale}
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: {scale}
      test_y_channel: false
"""


def render_arch_block(model: str, arch: dict) -> str:
    lines = [f'  type: {arch.get("type", model)}']
    for k, v in arch.items():
        if k == 'type':
            continue
        if isinstance(v, (list, tuple)):
            v = '[' + ', '.join(str(x) for x in v) + ']'
        elif isinstance(v, str):
            v = f"'{v}'"
        lines.append(f'  {k}: {v}')
    return '\n'.join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='实验总配置文件路径（默认自动查找）')
    p.add_argument('--data_root', default=None, help='默认取总配置的 paths.data')
    p.add_argument('--out', default=None, help='默认取总配置的 paths.configs')
    p.add_argument('--scale', type=int, nargs='+', default=None)
    p.add_argument('--protocols', nargs='+', default=None)
    p.add_argument('--models', nargs='+', default=None,
                   help='默认取总配置的 comparison + ablation + ours')
    return p.parse_args()


def main():
    args = parse_args()

    # ---- 读实验总配置（唯一事实来源）----
    cfg = None
    if _HAS_EXP_CFG:
        try:
            cfg = load_exp_cfg(args.config)
            print(f"[gen] 读取总配置: {cfg['_config_path']}")
        except FileNotFoundError as e:
            print(f'[gen] 警告: {e}')
            print('[gen] 回落到脚本内置默认值 —— 注意这可能与其它实验不一致！')
    else:
        print('[gen] 警告: 找不到 exp_config.py，使用内置默认值')

    if cfg:
        # 总配置里是相对路径（为了可移植）；生成 yml 时解析成本机绝对路径，
        # 因为 BasicSR 从 $BASICSR_ROOT 启动、按 CWD 解析 dataroot。
        P = paths_abs(cfg)
        args.data_root = args.data_root or P['data']
        args.out = args.out or P['configs']
        args.scale = args.scale or list(cfg['degradation']['scale'])
        args.protocols = args.protocols or list(cfg['degradation']['protocols'])
        args.models = args.models or (cfg['models']['comparison']
                                      + cfg['models']['ablation']
                                      + cfg['models']['ours'])
    args.data_root = args.data_root or 'data/holo'
    args.out = args.out or 'configs'
    args.scale = args.scale or [4]
    args.protocols = args.protocols or ['bicubic', 'physical']
    args.models = args.models or list(MODELS)

    common = build_common(cfg)
    out = Path(args.out)
    (out / 'train').mkdir(parents=True, exist_ok=True)
    (out / 'test').mkdir(parents=True, exist_ok=True)

    n = 0
    for model in args.models:
        if model not in MODELS:
            print(f'[gen] 跳过未知模型 {model}（可选: {list(MODELS)}）')
            continue
        spec = MODELS[model]
        arch_block = render_arch_block(model, spec['arch'])
        # DAT 仓库里 DAT 自带 DATModel（SRModel + 大图分块推理），其余用 SRModel
        model_type = 'DATModel' if model == 'DAT' else 'SRModel'
        for scale in args.scale:
            for protocol in args.protocols:
                ctx = dict(
                    model=model, model_type=model_type, protocol=protocol,
                    scale=scale, data_root=args.data_root,
                    arch_block=arch_block, note=spec['note'],
                    source=spec['source'], **common)
                (out / 'train' / f'{model}_{protocol}_x{scale}.yml').write_text(
                    TRAIN_TPL.format(**ctx), encoding='utf-8')
                (out / 'test' / f'{model}_{protocol}_x{scale}.yml').write_text(
                    TEST_TPL.format(**ctx), encoding='utf-8')
                n += 2

    print(f'[gen] 生成 {n} 个配置 -> {out}/train、{out}/test')
    print('[gen] 零接入（上游 BasicSR 自带，免放文件）: '
          'SRResNet / EDSR / RCAN / RRDBNet / SwinIR')
    print('[gen] 需接入 arch 文件: HAT / NAFNet / MambaIR（DAT 本仓库自带）')
    print(f'[gen] 模型清单: {", ".join(args.models)}')


if __name__ == '__main__':
    main()
