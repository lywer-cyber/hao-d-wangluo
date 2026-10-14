import importlib
import json
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.gradient_accumulation_steps = int(
            train_opt.get('gradient_accumulation_steps', 1))
        if self.gradient_accumulation_steps < 1:
            raise ValueError('gradient_accumulation_steps must be >= 1')
        effective_batch = int(train_opt.get('effective_batch_size_per_gpu', 0))
        micro_batch = int(self.opt['datasets']['train'].get(
            'batch_size_per_gpu', 0))
        if effective_batch < 1 or micro_batch < 1 or (
                effective_batch != micro_batch * self.gradient_accumulation_steps):
            raise ValueError(
                'effective_batch_size_per_gpu must equal '
                'micro_batch_size_per_gpu * gradient_accumulation_steps')
        if not bool(train_opt.get('loss_divide_by_accumulation_steps', False)):
            raise ValueError('loss_divide_by_accumulation_steps must be true')
        if not bool(train_opt.get('amp_grad_scaler', False)):
            raise ValueError('amp_grad_scaler must be true')
        clip_enabled = bool(train_opt.get('use_grad_clip', False)) or bool(
            train_opt.get('gradient_clip_enabled', False))
        self.grad_clip_norm = None
        if clip_enabled:
            clip_value = train_opt.get('gradient_clip_norm')
            if clip_value in (None, 'none', 'null') or float(clip_value) <= 0:
                raise ValueError('gradient_clip_norm must be positive when clipping is enabled')
            self.grad_clip_norm = float(clip_value)
        self.amp_enabled = bool(train_opt.get('amp_enabled', False)) and self.device.type == 'cuda'
        self.amp_dtype = str(train_opt.get('amp_dtype', 'float16')).lower()
        if self.amp_dtype != 'float16':
            raise ValueError('The unified experiment requires amp_dtype=float16')
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled,
            init_scale=float(train_opt.get('amp_init_scale', 65536.0)),
            growth_factor=float(train_opt.get('amp_growth_factor', 2.0)),
            backoff_factor=float(train_opt.get('amp_backoff_factor', 0.5)),
            growth_interval=int(train_opt.get('amp_growth_interval', 2000)))
        self._micro_step = 0
        self._best_selection = None
        best_path = osp.join(self.opt['path'].get('experiments_root', ''),
                             'best_checkpoint.json')
        if best_path and osp.isfile(best_path):
            try:
                with open(best_path, 'r', encoding='utf-8') as f:
                    self._best_selection = json.load(f)
            except (OSError, ValueError):
                self._best_selection = None

        self.ema_decay = float(train_opt.get('ema_decay', 0)) \
            if bool(train_opt.get('ema_enabled', False)) else 0.0
        self.ema_warmup_iters = int(train_opt.get('ema_warmup_iters', 0))
        self.loss_l1_weight = float(train_opt.get(
            'loss_l1_weight', train_opt.get('pixel_opt', {}).get('loss_weight', 1.0)))
        self.loss_gfl_weight = float(train_opt.get('loss_gfl_weight', 0.0))
        self.loss_hf_weight = float(train_opt.get('loss_hf_weight', 0.0))
        self.gfl_ramp_iterations = int(train_opt.get('gfl_ramp_iterations', 30000))
        self.gfl_radius_ref = float(train_opt.get('gfl_radius_ref', 0.125))
        self.hf_sigma_px = float(train_opt.get('hf_sigma_px', 1.8006))
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        self.optimizer_g.zero_grad(set_to_none=True)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def resume_training(self, resume_state):
        """Resume optimizer, scheduler, AMP scaler, and accumulation phase."""
        super().resume_training(resume_state)
        if 'grad_scaler' in resume_state:
            self.grad_scaler.load_state_dict(resume_state['grad_scaler'])
        self._micro_step = int(resume_state.get('micro_step', 0))

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def _frequency_loss(self, sr, gt, current_iter):
        _, _, height, width = sr.shape
        fy = torch.fft.fftshift(torch.fft.fftfreq(height, d=1.0, device=sr.device))
        fx = torch.fft.fftshift(torch.fft.fftfreq(width, d=1.0, device=sr.device))
        radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        radius = radius / self.gfl_radius_ref
        ramp = max(self.gfl_ramp_iterations, 1)
        rho = max(0.0, min(1.0, 1.0 - float(current_iter) / ramp))
        mask = radius >= rho
        if not bool(mask.any()):
            mask = torch.ones_like(mask, dtype=torch.bool)
        diff = torch.fft.fftshift(
            torch.fft.fft2(sr, dim=(-2, -1), norm='ortho')) - torch.fft.fftshift(
                torch.fft.fft2(gt, dim=(-2, -1), norm='ortho'))
        diff = diff.abs()
        denom = mask.sum().to(diff.dtype) * sr.shape[0] * sr.shape[1]
        return (diff * mask).sum() / denom.clamp_min(1.0)

    def _high_frequency_loss(self, sr, gt):
        radius = 4
        coords = torch.arange(-radius, radius + 1, device=sr.device,
                              dtype=sr.dtype)
        sigma = self.hf_sigma_px
        kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel.expand(sr.shape[1], 1, 2 * radius + 1, 2 * radius + 1)

        def deterministic_reflect_pad(value):
            # F.pad(..., mode='reflect') has no deterministic CUDA backward in
            # the pinned PyTorch runtime.  These slice/flip/concat operations
            # produce the identical reflect boundary values without invoking
            # reflection_pad2d_backward.
            def pad_dim(tensor, dim):
                size = tensor.shape[dim]
                if size <= radius:
                    raise ValueError('HF reflect padding requires image dimensions > radius')
                left = tensor.narrow(dim, 1, radius).flip(dim)
                right = tensor.narrow(dim, size - radius - 1, radius).flip(dim)
                return torch.cat((left, tensor, right), dim=dim)

            return pad_dim(pad_dim(value, -1), -2)

        def high_pass(value):
            blurred = F.conv2d(deterministic_reflect_pad(value),
                               kernel, groups=value.shape[1])
            return value - blurred

        return F.l1_loss(high_pass(sr), high_pass(gt), reduction='mean')

    def optimize_parameters(self, current_iter):
        # One call is one micro-batch; current_iter is one optimizer update.
        with torch.cuda.amp.autocast(
                enabled=self.amp_enabled, dtype=torch.float16):
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]
            self.output = preds[-1]

        # FFT and spatial losses are deliberately evaluated in float32, even
        # when the forward pass uses float16 autocast.
        with torch.cuda.amp.autocast(enabled=False):
            sr = self.output.float()
            gt = self.gt.float()
            l1 = sum(self.cri_pix(pred.float(), gt) for pred in preds)
            l_gfl = self._frequency_loss(sr, gt, current_iter)
            l_hf = self._high_frequency_loss(sr, gt)
            l_pix = (self.loss_l1_weight * l1 +
                     self.loss_gfl_weight * l_gfl +
                     self.loss_hf_weight * l_hf)

        loss_dict = OrderedDict(l_pix=l1.detach(), l_gfl=l_gfl.detach(),
                                l_hf=l_hf.detach(), l_total=l_pix.detach())
        # Every micro-batch contributes 1/accumulation_steps of the effective
        # batch loss.  GradScaler is stepped exactly once per optimizer update.
        loss_for_backward = l_pix / self.gradient_accumulation_steps
        self.grad_scaler.scale(loss_for_backward).backward()
        self._micro_step += 1
        did_update = self._micro_step % self.gradient_accumulation_steps == 0

        if did_update:
            if self.grad_clip_norm is not None:
                self.grad_scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip_norm)
            self.grad_scaler.step(self.optimizer_g)
            self.grad_scaler.update()
            self.optimizer_g.zero_grad(set_to_none=True)
            if self.ema_decay > 0:
                if self.ema_warmup_iters > 0:
                    warm_decay = self.ema_decay * (1.0 - torch.exp(
                        torch.tensor(-float(current_iter) /
                                     self.ema_warmup_iters)).item())
                else:
                    warm_decay = self.ema_decay
                self.model_ema(decay=warm_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
        return did_update

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ.get('LOCAL_RANK', '0') == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if self.opt.get('holo_protocol', False) and window_size:
            raise ValueError('Holographic validation forbids padding; val.window_size must be 0')

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            if self.opt.get('holo_protocol', False):
                if self.output.shape[-2:] != self.gt.shape[-2:]:
                    raise RuntimeError(
                        f'Native-resolution validation shape mismatch: '
                        f'SR={tuple(self.output.shape[-2:])}, '
                        f'HR={tuple(self.gt.shape[-2:])}')

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                if self.opt.get('holo_protocol', False):
                    # Validation and evaluate_holo.py intentionally share the
                    # same aggregate implementation and parameter names.
                    metric_opt = self.opt['val']['metrics']
                    psnr_opt = metric_opt.get('psnr', {})
                    ssim_opt = metric_opt.get('ssim', {})
                    metric_sr = (visuals['result'].clamp(0, 1)
                                 if not use_image else sr_img)
                    metric_gt = (visuals['gt'].clamp(0, 1)
                                 if not use_image else gt_img)
                    shared = metric_module.calculate_holo_metrics(
                        metric_sr, metric_gt,
                        crop_border=int(psnr_opt.get('crop_border', 4)),
                        psnr_data_range=float(psnr_opt.get('data_range', 1.0)),
                        ssim_data_range=float(ssim_opt.get('data_range', 1.0)),
                        ssim_window_size=int(ssim_opt.get('win_size', 11)),
                        ssim_sigma=float(ssim_opt.get('sigma', 1.5)),
                        ssim_gaussian_weights=bool(
                            ssim_opt.get('gaussian_weights', True)),
                        ssim_use_sample_covariance=bool(
                            ssim_opt.get('use_sample_covariance', False)),
                        ssim_channel_axis=int(ssim_opt.get('channel_axis', 2)),
                        ssim_channel_reduction=ssim_opt.get(
                            'channel_reduction', 'mean'))
                    for name in self.metric_results:
                        self.metric_results[name] += float(
                            shared[name.upper()])
                else:
                    opt_metric = deepcopy(self.opt['val']['metrics'])
                    if use_image:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(sr_img, gt_img, **opt_)
                    else:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
            if self.opt.get('is_train', False):
                self._update_best_checkpoint(current_iter)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def _metric_value(self, name):
        for key, value in self.metric_results.items():
            if key.lower() == name.lower():
                return float(value)
        return None

    def _update_best_checkpoint(self, current_iter):
        """Select checkpoints using validation PSNR, then SSIM only for ties."""
        val_opt = self.opt.get('val', {})
        if val_opt.get('checkpoint_selection_split', 'val') != 'val':
            raise ValueError('Checkpoint selection must use the val split')
        psnr = self._metric_value(val_opt.get('checkpoint_selection_metric', 'PSNR'))
        ssim = self._metric_value(val_opt.get('checkpoint_selection_tie_breaker', 'SSIM'))
        if psnr is None or ssim is None:
            raise ValueError('Validation must report PSNR and SSIM for checkpoint selection')
        tol = float(val_opt.get('checkpoint_selection_tie_tolerance', 1e-12))
        best = getattr(self, '_best_selection', None)
        better = best is None or psnr > best['PSNR'] + tol
        if best is not None and abs(psnr - best['PSNR']) <= tol:
            better = ssim > best['SSIM'] + tol
        if not better:
            return False

        self._best_selection = {'iter': int(current_iter), 'PSNR': psnr, 'SSIM': ssim}
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g_best', current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g_best', current_iter)
        record_path = osp.join(self.opt['path']['experiments_root'], 'best_checkpoint.json')
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(self._best_selection, f, indent=2)
        get_root_logger().info(
            f"New best checkpoint at iter {current_iter}: PSNR={psnr:.6f}, SSIM={ssim:.6f}")
        return True

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
