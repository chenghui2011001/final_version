#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4: Quantised full optimisation with channel perturbations.

This training stage continues from the Stage 3 checkpoint and keeps the
encoder/decoder/vocoder fully trainable while adding bitrate and MoE
regularisation terms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import soundfile as sf
import numpy as np
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Ensure project root is importable when running as a script
_THIS_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

# Use Stage3-consistent encoder (DualStream + MoE + FiLM)
from models.enhanced_aether_integration import AETHEREncoder
from models.maybe_useless.aether_fargan_decoder import AETHERFARGANDecoder
# 双头解码器和SSL工具
from models.semantic_augmented_aether_decoder import SemanticAugmentedAETHERDecoder
from models.ssl_utils import load_ssl_model
# Stage3语义模块（必需）
from models.semantic_extractor import create_semantic_extractor
from utils.fsk import TwoFSKModem
from training.losses import (
    balance_loss,
    compute_layered_loss,
    rate_loss,
    router_consistency_loss,
)
from training.acoustic_adversarial_loss import create_acoustic_adversarial_loss
from utils.real_data_loader import create_aether_data_loader, create_combined_data_loader
from utils.channel_sim import ChannelSimulator
from models.utils import build_csi_vec
try:
    from models.fargan_decoder import FARGANDecoder as _RefFARGANDecoder
except Exception:
    _RefFARGANDecoder = None
from models.fargan_components import GLU
try:
    from .stages import StageConfig, get_stage_config  # package mode
    from .wave_loss import fargan_wave_losses
except Exception:
    from training.pipeline.stages import StageConfig, get_stage_config  # script mode fallback
    from training.pipeline.wave_loss import fargan_wave_losses
from tqdm.auto import tqdm

# --------------------------
# Distributed training utilities
# --------------------------

def setup_distributed():
    """Initialize distributed training if using multiple processes"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Multi-node or multi-GPU distributed training
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Initialize process group
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        return True, rank, world_size, local_rank
    else:
        # Single node training
        return False, 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    """Check if this is the main process (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes for averaging"""
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

# --------------------------
# Stage3-style inspection utils
# --------------------------
def _print_feature_reconstruction_stats_36(pred_feats: torch.Tensor, orig_feats: torch.Tensor, global_step: int, batch_idx: int) -> None:
    """Stage3-style: compare reconstructed 36-dim FARGAN features against targets.
    Focus on ceps(0-17), f0(18), corr(19) and overall metrics (front-20).
    """
    with torch.no_grad():
        try:
            tqdm.write(f"\n========== 特征重建统计 (Step {global_step}, Batch {batch_idx}) ==========")
            # 0-17: Cepstral
            tqdm.write("--- 倒谱特征 (Dims 0-17) ---")
            for dim in range(18):
                p = pred_feats[:, :, dim].flatten()
                g = orig_feats[:, :, dim].flatten()
                tqdm.write(
                    f"  Dim[{dim:2d}] | Pred: mean={p.mean().item():+6.3f} std={p.std().item():6.3f} "
                    f"range=[{p.min().item():+6.3f}, {p.max().item():+6.3f}]\n"
                    f"         | Orig: mean={g.mean().item():+6.3f} std={g.std().item():6.3f} "
                    f"range=[{g.min().item():+6.3f}, {g.max().item():+6.3f}]"
                )

            # 18: F0 (DNN pitch)
            tqdm.write("\n--- F0/基频特征 (Dim 18) ---")
            pf0 = pred_feats[:, :, 18].flatten(); gf0 = orig_feats[:, :, 18].flatten()
            tqdm.write(
                f"  F0     | Pred: mean={pf0.mean().item():+6.3f} std={pf0.std().item():6.3f} "
                f"range=[{pf0.min().item():+6.3f}, {pf0.max().item():+6.3f}]\n"
                f"         | Orig: mean={gf0.mean().item():+6.3f} std={gf0.std().item():6.3f} "
                f"range=[{gf0.min().item():+6.3f}, {gf0.max().item():+6.3f}]"
            )
            # voiced ratio (approx)
            pv = (pf0 > -1.0).float().mean().item(); gv = (gf0 > -1.0).float().mean().item()
            tqdm.write(f"  Voice  | Pred: voiced={pv:.3f} unvoiced={1-pv:.3f}\n         | Orig: voiced={gv:.3f} unvoiced={1-gv:.3f}")

            # 19: Frame corr
            tqdm.write("\n--- 帧相关性特征 (Dim 19) ---")
            pc = pred_feats[:, :, 19].flatten(); gc = orig_feats[:, :, 19].flatten()
            tqdm.write(
                f"  Corr   | Pred: mean={pc.mean().item():+6.3f} std={pc.std().item():6.3f} "
                f"range=[{pc.min().item():+6.3f}, {pc.max().item():+6.3f}]\n"
                f"         | Orig: mean={gc.mean().item():+6.3f} std={gc.std().item():6.3f} "
                f"range=[{gc.min().item():+6.3f}, {gc.max().item():+6.3f}]"
            )

            # Overall metrics on front-20
            tqdm.write("\n--- 整体重建质量（前20维） ---")
            o_mse = F.mse_loss(pred_feats[:, :, :20], orig_feats[:, :, :20]).item()
            o_mae = F.l1_loss(pred_feats[:, :, :20], orig_feats[:, :, :20]).item()
            ce_mse = F.mse_loss(pred_feats[:, :, :18], orig_feats[:, :, :18]).item()
            f0_mse = F.mse_loss(pred_feats[:, :, 18:19], orig_feats[:, :, 18:19]).item()
            cr_mse = F.mse_loss(pred_feats[:, :, 19:20], orig_feats[:, :, 19:20]).item()
            tqdm.write(f"  Overall MSE: {o_mse:.6f}, MAE: {o_mae:.6f}")
            tqdm.write(f"  Cepstral MSE: {ce_mse:.6f}")
            tqdm.write(f"  F0 MSE: {f0_mse:.6f}")
            tqdm.write(f"  Correlation MSE: {cr_mse:.6f}")
            tqdm.write("=" * 65)
        except Exception:
            pass

def _print_acoustic20_comparison(pred20: torch.Tensor, tgt20: torch.Tensor, global_step: int, batch_idx: int) -> None:
    """Compare 20-dim acoustic features (dual-head) against target front-20.
    Prints ceps/f0/corr stats similar to Stage3.
    """
    with torch.no_grad():
        try:
            tqdm.write(f"\n---------- 20维声学特征对比 (Step {global_step}, Batch {batch_idx}) ----------")
            # 0-17
            for dim in range(18):
                p = pred20[:, :, dim].flatten(); g = tgt20[:, :, dim].flatten()
                tqdm.write(
                    f"  A20[{dim:2d}] | Pred: mean={p.mean().item():+6.3f} std={p.std().item():6.3f} "
                    f"range=[{p.min().item():+6.3f}, {p.max().item():+6.3f}]\n"
                    f"           | Tgt : mean={g.mean().item():+6.3f} std={g.std().item():6.3f} "
                    f"range=[{g.min().item():+6.3f}, {g.max().item():+6.3f}]"
                )
            # 18,19
            for dim, name in [(18, 'F0'), (19, 'Corr')]:
                p = pred20[:, :, dim].flatten(); g = tgt20[:, :, dim].flatten()
                tqdm.write(
                    f"  {name:<5} | Pred: mean={p.mean().item():+6.3f} std={p.std().item():6.3f} "
                    f"range=[{p.min().item():+6.3f}, {p.max().item():+6.3f}]\n"
                    f"         | Tgt : mean={g.mean().item():+6.3f} std={g.std().item():6.3f} "
                    f"range=[{g.min().item():+6.3f}, {g.max().item():+6.3f}]"
                )
            # Overall
            mse = F.mse_loss(pred20, tgt20).item(); mae = F.l1_loss(pred20, tgt20).item()
            tqdm.write(f"  A20 Overall MSE: {mse:.6f}, MAE: {mae:.6f}")
            tqdm.write("-" * 65)
        except Exception:
            pass

def _print_semantic_stats_16(sem_pred: torch.Tensor, sem_tgt: Optional[torch.Tensor], global_step: int, batch_idx: int) -> None:
    """Print 16-d semantic features statistics; if target provided, print side-by-side and overall MSE/MAE.
    This mirrors Stage3's semantic alignment inspection.
    """
    with torch.no_grad():
        try:
            tqdm.write(f"\n---------- 语义特征 (16D) (Step {global_step}, Batch {batch_idx}) ----------")
            for dim in range(sem_pred.size(-1)):
                p = sem_pred[:, :, dim].flatten()
                if sem_tgt is not None:
                    t = sem_tgt[:, :, dim].flatten()
                    tqdm.write(
                        f"  Sem[{dim:2d}] | Pred: mean={p.mean().item():+6.3f} std={p.std().item():6.3f} "
                        f"range=[{p.min().item():+6.3f}, {p.max().item():+6.3f}]\n"
                        f"           | Tgt : mean={t.mean().item():+6.3f} std={t.std().item():6.3f} "
                        f"range=[{t.min().item():+6.3f}, {t.max().item():+6.3f}]"
                    )
                else:
                    tqdm.write(
                        f"  Sem[{dim:2d}] | Pred: mean={p.mean().item():+6.3f} std={p.std().item():6.3f} "
                        f"range=[{p.min().item():+6.3f}, {p.max().item():+6.3f}]"
                    )
            if sem_tgt is not None:
                mse = F.mse_loss(sem_pred.float(), sem_tgt.float()).item()
                mae = F.l1_loss(sem_pred.float(), sem_tgt.float()).item()
                tqdm.write(f"  Semantic MSE: {mse:.6f}, MAE: {mae:.6f}")
            tqdm.write("-" * 65)
        except Exception:
            pass


def safe_print(msg: str, flush: bool = False) -> None:
    """Stage3-style safe printer that cooperates with tqdm progress bars."""
    # Only print from main process in distributed training
    if not is_main_process():
        return
    try:
        tqdm.write(str(msg))
    except Exception:
        print(msg)
    if flush:
        try:
            import sys as _sys
            _sys.stdout.flush()
        except Exception:
            pass


def _load_stage3_checkpoint(
    encoder: AETHEREncoder,
    decoder: AETHERFARGANDecoder,
    ckpt_path: Optional[Path],
) -> None:
    if ckpt_path is None:
        return
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage 3 checkpoint not found: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location="cpu")
    enc_sd = payload.get("encoder_state_dict")
    dec_sd = payload.get("decoder_state_dict")
    # --- Decoder compatibility mapping (Stage3→Stage4) ---
    # Older checkpoints may use a single-head refiner (refiner.out_proj). Map to new 3-head refiner.
    if isinstance(dec_sd, dict):
        try:
            ds = dict(dec_sd)
            # If old single head exists and new heads don't, create mapped entries
            if ('refiner.out_proj.weight' in ds or 'refiner.out_proj.bias' in ds) and \
               ('refiner.out_mu.weight' not in ds and hasattr(decoder, 'refiner') and hasattr(decoder.refiner, 'out_mu')):
                w = ds.get('refiner.out_proj.weight', None)
                b = ds.get('refiner.out_proj.bias', None)
                if w is not None:
                    ds['refiner.out_mu.weight'] = w
                if b is not None:
                    ds['refiner.out_mu.bias'] = b
                # Provide zero init for eps/logstd heads so y≈mu at init
                mu_w = decoder.refiner.out_mu.weight
                eps_w = decoder.refiner.out_eps.weight
                log_w = decoder.refiner.out_logstd.weight
                ds.setdefault('refiner.out_eps.weight', torch.zeros_like(eps_w, dtype=eps_w.dtype))
                ds.setdefault('refiner.out_eps.bias', torch.zeros_like(decoder.refiner.out_eps.bias, dtype=decoder.refiner.out_eps.bias.dtype))
                # Bias slightly negative so softplus is small; but eps=0 makes it moot
                ds.setdefault('refiner.out_logstd.weight', torch.zeros_like(log_w, dtype=log_w.dtype))
                ds.setdefault('refiner.out_logstd.bias', torch.full_like(decoder.refiner.out_logstd.bias, -5.0))
                # Remove legacy keys to avoid "unexpected" spam
                ds.pop('refiner.out_proj.weight', None)
                ds.pop('refiner.out_proj.bias', None)
                dec_sd = ds
        except Exception:
            pass
    if enc_sd:
        res_e = encoder.load_state_dict(enc_sd, strict=False)
        try:
            missing = getattr(res_e, 'missing_keys', [])
            unexpected = getattr(res_e, 'unexpected_keys', [])
            if missing or unexpected:
                safe_print(f"[Stage3→4] Encoder load: missing={len(missing)} unexpected={len(unexpected)}")
                if unexpected:
                    safe_print(f"  e.g. unexpected: {unexpected[:3]}")
                if missing:
                    safe_print(f"  e.g. missing: {missing[:3]}")
        except Exception:
            pass
    if dec_sd:
        # Pre-load diagnostics: fargan_core hit ratio
        try:
            total_fc = sum(1 for k in dec_sd.keys() if k.startswith('fargan_core.'))
            present_fc = sum(1 for k in dec_sd.keys() if k.startswith('fargan_core.') and any(
                k == kk or kk.endswith(k) for kk in decoder.state_dict().keys()
            ))
            if total_fc > 0:
                pct = 100.0 * present_fc / max(1, total_fc)
                safe_print(f"[Stage3→4] fargan_core keys in ckpt: {present_fc}/{total_fc} (~{pct:.1f}%)")
        except Exception:
            pass

        res_d = decoder.load_state_dict(dec_sd, strict=False)
        try:
            missing = getattr(res_d, 'missing_keys', [])
            unexpected = getattr(res_d, 'unexpected_keys', [])
            if missing or unexpected:
                safe_print(f"[Stage3→4] Decoder load: missing={len(missing)} unexpected={len(unexpected)}")
                if unexpected:
                    safe_print(f"  e.g. unexpected: {unexpected[:3]}")
                if missing:
                    safe_print(f"  e.g. missing: {missing[:3]}")
            # Coverage for fargan_core on the model side
            try:
                model_fc_total = sum(1 for k in decoder.state_dict().keys() if k.startswith('fargan_core.'))
                missing_fc = sum(1 for k in missing if k.startswith('fargan_core.'))
                matched_fc = model_fc_total - missing_fc
                if model_fc_total > 0:
                    pct_model = 100.0 * matched_fc / max(1, model_fc_total)
                    safe_print(f"[Stage3→4] fargan_core model coverage: {matched_fc}/{model_fc_total} (~{pct_model:.1f}%)")
            except Exception:
                pass
        except Exception:
            pass
        # Enforce identity calibration for params not present in Stage3 ckpt
        _set_decoder_identity_calib(decoder)


def _resume_stage4(
    encoder: AETHEREncoder,
    decoder: AETHERFARGANDecoder,
    optimizer: Optional[optim.Optimizer],
    ckpt_path: Optional[Path],
) -> Tuple[int, int, float]:
    """Resume from a Stage4 checkpoint if provided.

    Returns (start_epoch, start_global_step, best_loss).
    Handles checkpoints that may not include optimizer state (e.g., best.pth).
    """
    start_epoch = 1
    start_step = 0
    best_loss = float("inf")
    if ckpt_path is None:
        return start_epoch, start_step, best_loss
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location="cpu")
    enc_sd = payload.get("encoder_state_dict")
    dec_sd = payload.get("decoder_state_dict")
    if isinstance(enc_sd, dict):
        encoder.load_state_dict(enc_sd, strict=False)
    if isinstance(dec_sd, dict):
        decoder.load_state_dict(dec_sd, strict=False)
    # Optional optimizer resume
    if optimizer is not None and isinstance(payload.get('optimizer_state_dict'), dict):
        try:
            optimizer.load_state_dict(payload['optimizer_state_dict'])
        except Exception:
            pass
    # Epoch/step/best
    if isinstance(payload.get('epoch'), (int, float)):
        start_epoch = int(payload['epoch']) + 1
    if isinstance(payload.get('step'), (int, float)):
        start_step = int(payload['step'])
    if isinstance(payload.get('loss'), (int, float)):
        best_loss = float(payload['loss'])
    return start_epoch, start_step, best_loss

def _set_decoder_identity_calib(decoder: AETHERFARGANDecoder) -> None:
    """Force identity initialization for Stage4-only calibration params.
    This avoids amplitude collapse when loading Stage3 checkpoints (no such params).
    """
    try:
        modified = []
        if hasattr(decoder, 'ceps_gamma') and isinstance(decoder.ceps_gamma, torch.nn.Parameter):
            with torch.no_grad():
                decoder.ceps_gamma.data.fill_(1.0)
            modified.append('ceps_gamma')
        if hasattr(decoder, 'ceps_beta') and isinstance(decoder.ceps_beta, torch.nn.Parameter):
            with torch.no_grad():
                decoder.ceps_beta.data.zero_()
            modified.append('ceps_beta')
        if hasattr(decoder, 'feat_gamma_36') and isinstance(decoder.feat_gamma_36, torch.nn.Parameter):
            with torch.no_grad():
                decoder.feat_gamma_36.data.fill_(1.0)
            modified.append('feat_gamma_36')
        if hasattr(decoder, 'feat_beta_36') and isinstance(decoder.feat_beta_36, torch.nn.Parameter):
            with torch.no_grad():
                decoder.feat_beta_36.data.zero_()
            modified.append('feat_beta_36')
        if modified:
            safe_print(f"[Stage4] decoder calibration reset to identity: {modified}")
    except Exception as _e:
        safe_print(f"[WARN] Failed to reset decoder calibration: {_e}")

def _maybe_load_vocoder_ckpt(decoder: AETHERFARGANDecoder) -> None:
    """Optionally load a dedicated Stage2 vocoder checkpoint into decoder.fargan_core.
    Priority order:
      - env STAGE2_VOCODER_CKPT
      - checkpoints/checkpoints_stage2/checkpoints/best_model.pth (repo-relative)
    """
    import os
    paths = []
    p_env = os.environ.get('STAGE2_VOCODER_CKPT')
    if p_env:
        paths.append(p_env)
    try:
        guess = os.path.join(_ROOT_DIR, 'checkpoints', 'checkpoints_stage2', 'checkpoints', 'best_model.pth')
        paths.append(guess)
    except Exception:
        pass
    tried = []
    for p in paths:
        try:
            if not (p and os.path.exists(p)):
                continue
            import torch
            sd = torch.load(p, map_location='cpu')
            # Find a plausible state_dict
            cand = None
            for k in ('model_state_dict','state_dict','decoder_state_dict','vocoder_state_dict'):
                if k in sd and isinstance(sd[k], dict):
                    cand = sd[k]
                    break
            if cand is None:
                # maybe sd itself is a state dict
                if isinstance(sd, dict):
                    cand = sd
            if not isinstance(cand, dict):
                tried.append((p, 'no_state_dict'))
                continue
            # Extract fargan_core.* subset if present
            sub = {k: v for k, v in cand.items() if k.startswith('fargan_core.')}
            if not sub:
                # maybe nested under decoder.
                sub = {k.split('decoder.',1)[1]: v for k, v in cand.items() if k.startswith('decoder.fargan_core.')}
                sub = {k: v for k, v in sub.items() if k.startswith('fargan_core.')}
            if not sub:
                tried.append((p, 'no_fargan_core_keys'))
                continue
            res = decoder.fargan_core.load_state_dict(sub, strict=False)
            miss = getattr(res, 'missing_keys', [])
            unexp = getattr(res, 'unexpected_keys', [])
            safe_print(f"[Stage2→4] Loaded vocoder core from {p}: missing={len(miss)} unexpected={len(unexp)}")
            return
        except Exception as _e:
            tried.append((p, str(_e)))
            continue
    if tried:
        safe_print(f"[Stage2→4] No compatible vocoder ckpt loaded; tried: {tried}")

def _build_ref_vocoder_from_stage2(device: torch.device):
    """Construct a reference FARGANDecoder and load Stage2 ckpt if available.
    Returns the decoder instance or None.
    """
    if _RefFARGANDecoder is None:
        return None
    import os
    paths = []
    p_env = os.environ.get('STAGE2_VOCODER_CKPT')
    if p_env:
        paths.append(p_env)
    try:
        guess = os.path.join(_ROOT_DIR, 'checkpoints', 'checkpoints_stage2', 'checkpoints', 'best_model.pth')
        paths.append(guess)
    except Exception:
        pass
    tried = []
    for p in paths:
        try:
            if not (p and os.path.exists(p)):
                continue
            import torch
            sd = torch.load(p, map_location='cpu')
            cand = None
            for k in ('model_state_dict','state_dict','decoder_state_dict','vocoder_state_dict'):
                if k in sd and isinstance(sd[k], dict):
                    cand = sd[k]
                    break
            if cand is None and isinstance(sd, dict):
                cand = sd
            if not isinstance(cand, dict):
                tried.append((p, 'no_state_dict'))
                continue
            # Normalize prefixes (strip leading 'module.' or 'decoder.' or similar)
            mapped = {}
            for k, v in cand.items():
                kk = k
                for pref in ('module.', 'decoder.', 'model.', ''):
                    if pref and kk.startswith(pref):
                        kk = kk[len(pref):]
                mapped[kk] = v
            ref = _RefFARGANDecoder().to(device)
            res = ref.load_state_dict(mapped, strict=False)
            miss = getattr(res, 'missing_keys', [])
            unexp = getattr(res, 'unexpected_keys', [])
            safe_print(f"[Stage2→Ref] Loaded Stage2 vocoder into reference decoder: missing={len(miss)} unexpected={len(unexp)} from {p}")
            return ref
        except Exception as _e:
            tried.append((p, str(_e)))
            continue
    if tried:
        safe_print(f"[Stage2→Ref] Failed to build reference vocoder; tried: {tried}")
    return None


def _sum_grad_norm(named_params, include_key=None, exclude_key=None):
    total = 0.0
    n = 0
    for name, p in named_params:
        if p.grad is None:
            continue
        if include_key and include_key not in name:
            continue
        if exclude_key and exclude_key in name:
            continue
        g = torch.nan_to_num(p.grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        total += float(g.pow(2).sum().sqrt().item())
        n += 1
    return total, n


def _set_requires_grad(module: nn.Module, flag: bool, name_hint: str = "") -> None:
    for p in module.parameters():
        p.requires_grad = flag


def train_one_epoch(
    encoder: AETHEREncoder,
    decoder: AETHERFARGANDecoder,
    loader,
    device: torch.device,
    optimizer: optim.Optimizer,
    alpha_feat: float,
    alpha_wave: float,
    stage_cfg: StageConfig,
    current_step: int,
    use_2fsk: bool = False,
    fsk_modem: TwoFSKModem | None = None,
    fsk_bits_per_frame: int = 4,
    log_interval: int = 50,
    epoch_id: int = 1,
    num_epochs: int = 1,
    selected_csi_keys: Optional[List[str]] = None,
    save_every_steps: int = 0,
    out_dir: Optional[Path] = None,
    val_audio_interval: int = 0,
    val_audio_seconds: float = 0.0,
    val_audio_deemph: float = 0.0,
    wave_train_weights: Optional[Dict[str, float]] = None,
    scaler: Optional[GradScaler] = None,
    # 对抗损失相关参数
    use_adversarial_loss: bool = False,
    acoustic_adv_loss = None,
    disc_optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[Dict[str, float], float]:
    encoder.train()
    decoder.train()

    total = 0.0
    items = 0

    chan_sim = ChannelSimulator()

    # Freeze/unfreeze guards (persist across calls)
    if not hasattr(train_one_epoch, '_frozen'):  # type: ignore
        train_one_epoch._frozen = {
            'film': False,
            'moe': False,
            'dec': False,
        }

    progress = tqdm(
        enumerate(loader), total=len(loader), leave=False,
        desc=f"ep{epoch_id}/{num_epochs} JSCC"
    )
    for batch_idx, batch in progress:
        # Advance a global-like training step so FiLM/post schedules progress
        step = current_step + batch_idx
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        audio = batch["audio"].to(device, non_blocking=True)
        csi = batch.get("csi")
        if isinstance(csi, dict):
            csi = {k: v.to(device) for k, v in csi.items()}
        else:
            csi = {}

        optimizer.zero_grad(set_to_none=True)

        # Build simulated CSI for FiLM and apply matching channel on z
        B, T, _ = x.shape
        chan_type = stage_cfg.channel_type
        # SNR schedule: hi→lo with small sampling window
        snr_hi_db = float(getattr(train_one_epoch, 'snr_hi_db', 15.0))
        snr_lo_db = float(getattr(train_one_epoch, 'snr_lo_db', -5.0))
        snr_ramp_steps = int(getattr(train_one_epoch, 'snr_ramp_steps', 2000))
        snr_window_db = float(getattr(train_one_epoch, 'snr_window_db', 4.0))
        if snr_ramp_steps > 0:
            s_ratio = min(1.0, float(step) / float(snr_ramp_steps))
        else:
            s_ratio = 1.0
        center = (1.0 - s_ratio) * snr_hi_db + s_ratio * snr_lo_db
        half = max(0.0, snr_window_db * 0.5)
        lo_b = min(snr_lo_db, snr_hi_db)
        hi_b = max(snr_lo_db, snr_hi_db)
        snr_min = max(center - half, lo_b)
        snr_max = min(center + half, hi_b)
        csi_sim, amp_t, snr_db_t = chan_sim.sample_csi(B, T, channel=chan_type,
                                                       snr_min_db=snr_min, snr_max_db=snr_max)
        # Ensure ChannelSimulator outputs live on the model device
        try:
            if isinstance(csi_sim, dict):
                csi_sim = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in csi_sim.items()}
            if isinstance(amp_t, torch.Tensor):
                amp_t = amp_t.to(device)
            if isinstance(snr_db_t, torch.Tensor):
                snr_db_t = snr_db_t.to(device)
        except Exception:
            pass
        # Replace/extend CSI for FiLM: pass CSI dict (optionally filtered by --csi-keys)
        csi_cond = csi_sim
        if selected_csi_keys is not None:
            try:
                csi_cond = {k: v for k, v in csi_sim.items() if k in selected_csi_keys}
            except Exception:
                pass

        # After a configured step, force CSI proxies to randomise strongly
        try:
            rand_start = int(getattr(train_one_epoch, '_csi_random_start_step', 5000))
            rand_mode  = str(getattr(train_one_epoch, '_csi_random_mode', 'bimodal'))
        except Exception:
            rand_start, rand_mode = 5000, 'bimodal'
        if step >= rand_start and isinstance(csi_cond, dict):
            try:
                # Prepare random tensors on device
                dev = device
                Bv = B
                # 1) snr_proxy (dB): uniform in [-5, +5]
                snr_rand = (torch.rand(Bv, device=dev) * 10.0) - 5.0  # [-5,5]
                # 2) time_selectivity/freq_selectivity/los_ratio in [0,1]
                if rand_mode == 'uniform':
                    ts_rand  = torch.rand(Bv, device=dev)
                    fs_rand  = torch.rand(Bv, device=dev)
                    los_rand = torch.rand(Bv, device=dev)
                else:  # 'bimodal': switch between good/bad modes
                    m = torch.rand(Bv, device=dev) > 0.5  # True=bad, False=good
                    # time/freq selectivity: good(0.0~0.2) vs bad(0.8~1.0)
                    ts_good = torch.rand(Bv, device=dev) * 0.2
                    ts_bad  = 0.8 + torch.rand(Bv, device=dev) * 0.2
                    fs_good = torch.rand(Bv, device=dev) * 0.2
                    fs_bad  = 0.8 + torch.rand(Bv, device=dev) * 0.2
                    ts_rand = torch.where(m, ts_bad, ts_good)
                    fs_rand = torch.where(m, fs_bad, fs_good)
                    # los_ratio: good(0.7~1.0) vs bad(0.0~0.3)
                    los_good = 0.7 + torch.rand(Bv, device=dev) * 0.3
                    los_bad  = torch.rand(Bv, device=dev) * 0.3
                    los_rand = torch.where(m, los_bad, los_good)

                # Override existing keys if present; otherwise, insert for encoder FiLM
                if 'snr_proxy' in csi_cond:
                    csi_cond['snr_proxy'] = snr_rand
                else:
                    csi_cond.setdefault('snr_proxy', snr_rand)
                if 'time_selectivity' in csi_cond:
                    csi_cond['time_selectivity'] = ts_rand
                else:
                    csi_cond.setdefault('time_selectivity', ts_rand)
                if 'freq_selectivity' in csi_cond:
                    csi_cond['freq_selectivity'] = fs_rand
                else:
                    csi_cond.setdefault('freq_selectivity', fs_rand)
                if 'los_ratio' in csi_cond:
                    csi_cond['los_ratio'] = los_rand
                else:
                    csi_cond.setdefault('los_ratio', los_rand)
            except Exception:
                pass
        # Monkey-patch build_csi_vec path by passing through csi_dict internally
        # Here we override csi to just carry the tensors we want; encoder will rebuild vec to target dim
        # (AETHEREncoder uses models.utils.build_csi_vec)
        # Optional warm-ups (freezing)
        freeze_film_steps = int(getattr(train_one_epoch, 'freeze_film_steps', 0))
        freeze_moe_steps  = int(getattr(train_one_epoch, 'freeze_moe_steps', 0))
        freeze_dec_steps  = int(getattr(train_one_epoch, 'freeze_decoder_steps', 0))

        # Film warm-up: disable FiLM by not providing CSI to encoder/decoder
        revival_active = (step < freeze_film_steps)
        if revival_active:
            csi_for_enc = None
        else:
            csi_for_enc = csi_cond

        # MoE warm-up: freeze encoder.moe params for initial steps
        if freeze_moe_steps > 0:
            if step == 0 and not train_one_epoch._frozen['moe'] and hasattr(encoder, 'moe') and encoder.moe is not None:
                _set_requires_grad(encoder.moe, False, 'moe')
                train_one_epoch._frozen['moe'] = True
                safe_print(f"[Warmup] Freeze MoE for first {freeze_moe_steps} steps")
            if step == freeze_moe_steps and train_one_epoch._frozen['moe'] and hasattr(encoder, 'moe') and encoder.moe is not None:
                _set_requires_grad(encoder.moe, True, 'moe')
                train_one_epoch._frozen['moe'] = False
                safe_print("[Warmup] Unfreeze MoE")

        # Decoder (wave head) warm-up: freeze decoder params for initial steps
        if freeze_dec_steps > 0:
            if step == 0 and not train_one_epoch._frozen['dec']:
                _set_requires_grad(decoder, False, 'decoder')
                train_one_epoch._frozen['dec'] = True
                safe_print(f"[Warmup] Freeze decoder for first {freeze_dec_steps} steps")
            if step == freeze_dec_steps and train_one_epoch._frozen['dec']:
                _set_requires_grad(decoder, True, 'decoder')
                train_one_epoch._frozen['dec'] = False
                safe_print("[Warmup] Unfreeze decoder")

        # Forward pass with mixed precision
        try:
            from torch.amp import autocast as new_autocast
            amp_dtype = getattr(train_one_epoch, '_amp_dtype', torch.float16)
            autocast_context = new_autocast('cuda', enabled=(scaler is not None), dtype=amp_dtype)
        except ImportError:
            # Older API; dtype is accepted as 'dtype' in recent torch.cuda.amp
            try:
                amp_dtype = getattr(train_one_epoch, '_amp_dtype', torch.float16)
                autocast_context = autocast(enabled=(scaler is not None), dtype=amp_dtype)
            except TypeError:
                autocast_context = autocast(enabled=(scaler is not None))

        with autocast_context:
            # --- Revival alignment with Stage3 (optional, when revival_active) ---
            # Temporarily disable latent quantization and decoder-side FiLM to match Stage3 init audio
            enc_q_prev = getattr(encoder, 'use_quantization', None)
            enc_film_prev = getattr(encoder, 'use_film', None)
            dec_film_prev = None
            try:
                if revival_active:
                    if enc_q_prev is not None:
                        encoder.use_quantization = False
                    # 强制关闭编码端FiLM（不仅仅是不传CSI），确保复苏期与Stage3一致
                    if enc_film_prev is not None:
                        encoder.use_film = False
                    if hasattr(decoder, 'refiner') and hasattr(decoder.refiner, 'use_film'):
                        dec_film_prev = bool(decoder.refiner.use_film)
                        decoder.refiner.use_film = False
                    # Enable Stage3 compatibility mode for decoder refiner
                    if hasattr(decoder, 'refiner'):
                        setattr(decoder.refiner, '_stage3_compat_mode', True)
                    # 复苏期：FARGANCore用eval，避免训练态噪声注入（与Stage3一致）
                    prev_mode_fc = None
                    if hasattr(decoder, 'fargan_core'):
                        prev_mode_fc = decoder.fargan_core.training
                        decoder.fargan_core.eval()
                    # 复苏期：FARGANSub 使用 Stage3 严格输出激活（tanh）
                    prev_stage3_strict = None
                    try:
                        if hasattr(decoder, 'fargan_core') and hasattr(decoder.fargan_core, 'sig_net'):
                            prev_stage3_strict = bool(getattr(decoder.fargan_core.sig_net, '_stage3_strict', False))
                            setattr(decoder.fargan_core.sig_net, '_stage3_strict', True)
                    except Exception:
                        prev_stage3_strict = None
                    # 复苏期：确保解码端特征校准为单位映射（避免随机参数影响音色）
                    try:
                        _set_decoder_identity_calib(decoder)
                    except Exception:
                        pass
                    # 复苏期：全局禁用FARGAN量化噪声，并将GLU恒等（仅预览）
                    # 复苏期：全局禁用FARGAN量化噪声已移除（函数不存在）
                    prev_glu_flags = []
                    try:
                        if hasattr(decoder, 'fargan_core') and hasattr(decoder.fargan_core, 'sig_net'):
                            for m in decoder.fargan_core.sig_net.modules():
                                if isinstance(m, GLU):
                                    prev_glu_flags.append((m, bool(getattr(m, '_revival_identity', False))))
                                    setattr(m, '_revival_identity', True)
                    except Exception:
                        prev_glu_flags = []
            except Exception:
                pass
            # Pass training_step to enable FiLM strength ramping and other schedules
            z, logs = encoder(x, csi_for_enc, training_step=step)

            # Check for NaN in encoder output
            if torch.isnan(z).any() or torch.isinf(z).any():
                print(f"⚠️ NaN/Inf detected in encoder output z at step {step}")
                print(f"  z range: [{z.min():.6f}, {z.max():.6f}]")
                print(f"  z shape: {z.shape}")
                z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
            if use_2fsk and fsk_modem is not None:
                with torch.no_grad():
                    bits, meta = fsk_modem.bits_from_z(z.detach(), bits_per_frame=fsk_bits_per_frame)
                    wave = fsk_modem.modulate(bits)
                    # AWGN using batch-mean SNR
                    snr_db = csi_sim['snr_db'].detach().cpu()
                    for b in range(wave.size(0)):
                        snr_lin = float(torch.pow(10.0, snr_db[b] / 20.0))
                        std = float(wave[b].std().item() / (snr_lin + 1e-6))
                        wave[b] += torch.randn_like(wave[b]) * std
                    bits_hat = fsk_modem.demodulate(wave)
                    z = fsk_modem.z_from_bits(bits_hat, meta).to(device)
            elif stage_cfg.apply_channel:
                # Apply channel only after start step (revival skips channel)
                ch_start = int(getattr(train_one_epoch, '_channel_start_step', 0))
                if step >= ch_start:
                    z = chan_sim.apply(z, amp_t, snr_db_t)

            # Update decoder-side residual MoE with start gating (revival skips dec_moe)
            try:
                if hasattr(decoder, 'dec_moe') and decoder.dec_moe is not None:
                    dm_start = int(getattr(train_one_epoch, '_dec_moe_start_step', 0))
                    decoder.enable_dec_moe = bool(step >= dm_start)
                    if hasattr(decoder.dec_moe, 'set_training_step'):
                        decoder.dec_moe.set_training_step(step)
            except Exception:
                pass
            # SNR-aware period smoothing: disable during revival to match Stage3
            if not revival_active:
                try:
                    snr_mean_for_smooth = None
                    if isinstance(csi_cond, dict):
                        if 'snr_db' in csi_cond:
                            snr_mean_for_smooth = float(csi_cond['snr_db'].detach().mean().item())
                        elif 'snr_proxy' in csi_cond:
                            snr_mean_for_smooth = float(csi_cond['snr_proxy'].detach().mean().item())
                    if snr_mean_for_smooth is not None and hasattr(decoder, 'period_smooth_ks'):
                        decoder.period_smooth_ks = 3 if snr_mean_for_smooth < 5.0 else 1
                except Exception:
                    pass

            # Revival: provide warm-up pre frames to vocoder like Stage3 to avoid startup buzz
            if revival_active:
                try:
                    nb_pre = 2
                    a = audio
                    if a.dim() == 3:
                        a = a.squeeze(1)
                    pre = a[:, : nb_pre * 160].contiguous()
                    dec_csi = {'fargan_pre': pre}
                except Exception:
                    dec_csi = None
            else:
                dec_csi = csi_for_enc

            def _normalize_decoder_output(out):
                feats_, wav_ = None, None
                if isinstance(out, (tuple, list)):
                    if len(out) >= 1:
                        feats_ = out[0]
                    if len(out) >= 2:
                        wav_ = out[1]
                elif isinstance(out, dict):
                    # SemanticAugmentedAETHERDecoder returns a dict
                    feats_ = out.get('features', out.get('feats', out.get('y', None)))
                    wav_ = out.get('wave', None)
                else:
                    feats_ = out
                    wav_ = None
                return feats_, wav_

            # 检查是否使用语义增强解码器
            use_semantic_decoder = hasattr(decoder, 'get_semantic_info')

            if use_semantic_decoder:
                # 语义增强模式：直接在后面的分支中处理，这里只做基础前向用于兼容性
                if revival_active:
                    try:
                        from torch.amp import autocast as _ab_autocast
                        with _ab_autocast('cuda', enabled=False):
                            out = decoder(z.float(), dec_csi, return_wave=False)  # 不需要波形，减少计算
                    except Exception:
                        from torch.amp import autocast as _ab_autocast
                        with _ab_autocast('cuda', enabled=False):
                            out = decoder(z.float(), dec_csi, return_wave=False)
                else:
                    out = decoder(z, dec_csi, return_wave=False)

                # 暂时从基础解码器获取特征，后面会被融合特征替换
                feats, wav = _normalize_decoder_output(out)
                wav = None  # 将在语义增强分支中重新计算
            else:
                # 传统模式：正常前向传播
                if revival_active:
                    try:
                        from torch.amp import autocast as _ab_autocast
                        with _ab_autocast('cuda', enabled=False):
                            out = decoder(z.float(), dec_csi, return_wave=True, target_len=audio.size(-1))
                    except Exception:
                        from torch.amp import autocast as _ab_autocast
                        with _ab_autocast('cuda', enabled=False):
                            out = decoder(z.float(), dec_csi, return_wave=True, target_len=audio.size(-1))
                else:
                    out = decoder(z, dec_csi, return_wave=True, target_len=audio.size(-1))

                feats, wav = _normalize_decoder_output(out)
            # Restore flags after forward
            try:
                if revival_active:
                    if enc_q_prev is not None:
                        encoder.use_quantization = enc_q_prev
                    if enc_film_prev is not None:
                        encoder.use_film = enc_film_prev
                    if dec_film_prev is not None and hasattr(decoder, 'refiner'):
                        decoder.refiner.use_film = dec_film_prev
                    # 恢复FARGANCore运行模式
                    if 'prev_mode_fc' in locals() and prev_mode_fc is not None and hasattr(decoder, 'fargan_core'):
                        decoder.fargan_core.train(prev_mode_fc)
                    # 恢复FARGANSub激活模式
                    try:
                        if 'prev_stage3_strict' in locals() and prev_stage3_strict is not None and hasattr(decoder, 'fargan_core') and hasattr(decoder.fargan_core, 'sig_net'):
                            setattr(decoder.fargan_core.sig_net, '_stage3_strict', prev_stage3_strict)
                    except Exception:
                        pass
                    # 复苏期：恢复GLU与量化噪声开关
                    try:
                        if 'prev_glu_flags' in locals():
                            for m, old in prev_glu_flags:
                                setattr(m, '_revival_identity', old)
                    except Exception:
                        pass
                    # 恢复量化噪声已移除（函数不存在）
                else:
                    # Ensure Stage3 compat mode is disabled when not in revival
                    if hasattr(decoder, 'refiner'):
                        setattr(decoder.refiner, '_stage3_compat_mode', False)
            except Exception:
                pass
            feats = feats[:, : y.size(1), :]

            # 🔧 移除recon_base计算，因为已由acoustic_loss和semantic_loss替代
            # 可选的layered loss（如果启用）
            layered_comp = 0.0
            layered_loss = torch.tensor(0.0, device=device, dtype=feats.dtype)
            if stage_cfg.layered_enabled(step):
                layered_loss, _, _ = compute_layered_loss(
                    feats, y, current_step=step, feature_spec_type="fargan"
                )
                layered_comp = float(layered_loss.detach().item())

            # 初始化分支损失为0，确保异常/非双头路径下有定义
            semantic_loss = torch.tensor(0.0, device=device, dtype=feats.dtype)
            acoustic_loss = torch.tensor(0.0, device=device, dtype=feats.dtype)

            # 🔥 CRITICAL: Execute semantic augmentation BEFORE wave_loss to ensure wav is not None
            if use_semantic_decoder:
                try:
                    # 🔥 关键修复：直接使用语义增强模式获取融合后的波形和特征
                    # 这样FARGAN就直接使用融合后的特征合成波形
                    decoder_outputs = decoder(z, dec_csi, enable_semantic_output=True, return_wave=True, target_len=audio.size(-1))

                    # 提取融合后的特征和波形
                    acoustic_features = decoder_outputs['acoustic_features']   # [B, T, 20] 融合后A20
                    acoustic_raw = decoder_outputs['acoustic_raw']             # [B, T, 20] 预融合20维
                    semantic_features = decoder_outputs['semantic_features']   # [B, T, 16]
                    enhanced_features_36d = decoder_outputs['features_36d']    # [B, T, 36] 融合后36维

                    # 🚀 使用基于融合特征合成的波形
                    if 'wave' in decoder_outputs and decoder_outputs['wave'] is not None:
                        wav = decoder_outputs['wave']  # 基于融合特征的波形
                        if step % 20 == 0:  # 减少打印频率
                            print(f"[INFO] Using enhanced wave synthesis (shape: {wav.shape})")

                        # 可选：对比原始波形和融合波形（仅用于调试，长度对齐到最小值）
                        if 'wave_original' in decoder_outputs and step % 100 == 0:
                            try:
                                wav_original = decoder_outputs['wave_original']
                                w = wav if wav.dim() == 2 else wav.squeeze(1)
                                wo = wav_original if wav_original.dim() == 2 else wav_original.squeeze(1)
                                L = min(w.size(-1), wo.size(-1))
                                wave_diff_rms = torch.sqrt(torch.mean((w[..., :L] - wo[..., :L]) ** 2))
                                print(f"[DEBUG] Wave difference RMS: {wave_diff_rms.item():.6f}")
                            except Exception:
                                pass
                    else:
                        print(f"[WARNING] Enhanced wave synthesis not available, falling back to standard synthesis")
                        # 回退到标准合成
                        if revival_active:
                            try:
                                from torch.amp import autocast as _ab_autocast
                                with _ab_autocast('cuda', enabled=False):
                                    fallback_out = decoder(z.float(), dec_csi, return_wave=True, target_len=audio.size(-1))
                            except Exception:
                                fallback_out = decoder(z.float(), dec_csi, return_wave=True, target_len=audio.size(-1))
                        else:
                            fallback_out = decoder(z, dec_csi, return_wave=True, target_len=audio.size(-1))
                        _, wav = _normalize_decoder_output(fallback_out)

                    # 确保wav不为None
                    if wav is None:
                        print(f"[ERROR] Wave synthesis failed, generating zero wave")
                        B, T = enhanced_features_36d.shape[:2]
                        target_wav_len = audio.size(-1) if audio is not None else T * 160  # 假设160倍上采样
                        wav = torch.zeros(B, target_wav_len, device=enhanced_features_36d.device, dtype=enhanced_features_36d.dtype)

                    # 更新feats为融合后的36维特征，并标记来源
                    feats = enhanced_features_36d  # 替换原始feats
                    try:
                        train_one_epoch._feat_source = 'enhanced_36d'
                    except Exception:
                        pass

                    # 计算声学损失和语义损失（在这里计算，避免重复）
                    acoustic_target = y[..., :20]  # [B, T, 20]

                    if use_adversarial_loss and acoustic_adv_loss is not None:
                        # 对抗损失（生成器部分）
                        acoustic_loss, acoustic_metrics = acoustic_adv_loss.compute_generator_loss(
                            acoustic_features, acoustic_target
                        )
                        acoustic_loss = acoustic_loss * float(getattr(train_one_epoch, '_alpha_acoustic', 1.0))

                        # 添加融合正则项：防止A20偏离原始20维太远
                        fusion_reg_weight = float(getattr(train_one_epoch, '_fusion_reg_weight', 0.1))
                        if fusion_reg_weight > 0.0:
                            fusion_regularizer = F.mse_loss(acoustic_features, acoustic_raw) * fusion_reg_weight
                            acoustic_loss = acoustic_loss + fusion_regularizer
                            acoustic_metrics['fusion_regularizer'] = fusion_regularizer.item()
                            acoustic_metrics['fusion_reg_weight'] = fusion_reg_weight

                        # 记录对抗损失指标
                        if step % 20 == 0:
                            print(f"🎯 Acoustic Adversarial Metrics (Step {step}):")
                            for k, v in acoustic_metrics.items():
                                print(f"  {k}: {v:.6f}")
                    else:
                        # 🔥 自适应维度保护：加权L1损失
                        # 维度重要性权重：能量维度权重更高
                        dim_weights = torch.tensor([
                            3.0,  # Dim 0: 能量维度，最重要
                            2.0, 2.0, 1.5, 1.5, 1.5,  # Dim 1-5: 低频倒谱
                            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Dim 6-11: 中频倒谱
                            0.8, 0.8, 0.8, 0.8, 0.8, 0.8,  # Dim 12-17: 高频倒谱
                            2.5,  # Dim 18: F0，重要
                            1.2   # Dim 19: 相关性
                        ], device=acoustic_features.device).view(1, 1, 20)

                        weighted_diff = torch.abs(acoustic_features - acoustic_target) * dim_weights
                        acoustic_loss = weighted_diff.mean() * float(getattr(train_one_epoch, '_alpha_acoustic', 1.0))

                        # 添加融合正则项：防止A20偏离原始20维太远
                        fusion_reg_weight = float(getattr(train_one_epoch, '_fusion_reg_weight', 0.1))
                        if fusion_reg_weight > 0.0:
                            fusion_regularizer = F.mse_loss(acoustic_features, acoustic_raw) * fusion_reg_weight
                            acoustic_loss = acoustic_loss + fusion_regularizer

                    # 计算语义损失（启用 cosine/mse/infonce/cosine+infoce + 波形级约束 + 20→16蒸馏）
                    try:
                        sem_loss_type = str(getattr(train_one_epoch, '_semantic_loss_type', 'cosine'))
                        alpha_semantic = float(getattr(train_one_epoch, '_alpha_semantic', 0.3))
                        # 渐进式权重：前10k步逐渐加强监督
                        train_progress = min(1.0, step / 10000.0)
                        sem_scale = alpha_semantic * (0.1 + 0.9 * train_progress)

                        teacher_mode = str(getattr(train_one_epoch, '_semantic_teacher', 'ssl'))
                        ssl_teacher = getattr(train_one_epoch, '_ssl_teacher', None)
                        sem_ext = getattr(train_one_epoch, '_semantic_extractor', None)
                        temp = float(getattr(train_one_epoch, '_semantic_temperature', 0.8))
                        cw = float(getattr(train_one_epoch, '_semantic_cosine_weight', 0.3))
                        iw = float(getattr(train_one_epoch, '_semantic_infonce_weight', 0.7))
                        wwave = float(getattr(train_one_epoch, '_semantic_wave_weight', 0.3))
                        wdist = float(getattr(train_one_epoch, '_semantic_distill_weight', 0.5))

                        semantic_loss = torch.tensor(0.0, device=device, dtype=feats.dtype)

                        if teacher_mode == 'ssl' and ssl_teacher is not None:
                            # 使用SSL Teacher：调用解码器的语义损失聚合（包含投影/InfoNCE/波形级/蒸馏）
                            try:
                                with torch.no_grad():
                                    layer_idx = int(getattr(train_one_epoch, '_semantic_ssl_layer', 9))
                                    ssl_out = ssl_teacher(audio.detach(), output_hidden_states=True)
                                    ssl_feats = ssl_out.hidden_states[layer_idx]  # [B, T_ssl, D]

                                # 适配器：将SSL wrapper封装为返回[B, T_ssl, D]的提取器（供wave级约束使用）
                                class _SSLAdapter(torch.nn.Module):
                                    def __init__(self, wrapper, layer):
                                        super().__init__()
                                        self.w = wrapper
                                        self.layer = layer
                                    def forward(self, a: torch.Tensor):
                                        out = self.w(a, output_hidden_states=True)
                                        return out.hidden_states[self.layer]

                                ssl_adapter = _SSLAdapter(ssl_teacher, layer_idx).to(device)

                                # 20→16蒸馏特征（若存在）
                                distill_feat = decoder_outputs.get('acoustic_semantic_distill', None)

                                sem_dec_loss, _sem_metrics = decoder.compute_semantic_loss(
                                    semantic_features,
                                    ssl_feats,
                                    loss_type=sem_loss_type,
                                    wave_gt=audio,
                                    wave_rec=wav if wav is not None else None,
                                    ssl_extractor=ssl_adapter,
                                    acoustic_semantic_distill=distill_feat,
                                    distill_weight=wdist,
                                )
                                semantic_loss = sem_dec_loss * sem_scale
                            except Exception as _se:
                                print(f"⚠️ decoder-side semantic loss failed: {_se}")
                                # 回退到简单的cosine对齐（使用语义提取器）
                                if sem_ext is not None:
                                    with torch.no_grad():
                                        sem_tgt = sem_ext(audio.detach(), target_frames=semantic_features.size(1))
                                    sp = F.normalize(semantic_features.float(), dim=-1)
                                    st = F.normalize(sem_tgt.float(), dim=-1)
                                    semantic_loss = (1.0 - (sp * st).sum(dim=-1).mean()) * sem_scale

                        else:
                            # Stage3风格Teacher：使用16维语义提取器
                            if sem_ext is not None:
                                with torch.no_grad():
                                    sem_tgt = sem_ext(audio.detach(), target_frames=semantic_features.size(1))  # [B,T,16]

                                if sem_loss_type == 'cosine':
                                    sp = F.normalize(semantic_features.float(), dim=-1)
                                    st = F.normalize(sem_tgt.float(), dim=-1)
                                    base = 1.0 - (sp * st).sum(dim=-1).mean()
                                elif sem_loss_type == 'mse':
                                    base = F.mse_loss(semantic_features.float(), sem_tgt.float())
                                elif sem_loss_type in ('infonce', 'cosine+infoce'):
                                    sp = F.normalize(semantic_features.float(), dim=-1).reshape(-1, semantic_features.size(-1))
                                    st = F.normalize(sem_tgt.float(), dim=-1).reshape(-1, sem_tgt.size(-1))
                                    logits = (sp @ st.t()) / max(1e-6, temp)
                                    targets = torch.arange(sp.size(0), device=logits.device)
                                    infonce = F.cross_entropy(logits, targets)
                                    if sem_loss_type == 'infonce':
                                        base = infonce
                                    else:
                                        cos_part = 1.0 - (F.normalize(semantic_features.float(), dim=-1) * F.normalize(sem_tgt.float(), dim=-1)).sum(dim=-1).mean()
                                        base = cw * cos_part + iw * infonce
                                else:
                                    base = F.mse_loss(semantic_features.float(), sem_tgt.float())

                                semantic_loss = base * sem_scale

                                # 波形级语义约束（基于16维提取器）
                                try:
                                    if wwave > 0.0 and wav is not None:
                                        with torch.no_grad():
                                            sem_gt_w = sem_ext(audio.detach(), target_frames=semantic_features.size(1))
                                            sem_rec_w = sem_ext(wav.detach(), target_frames=semantic_features.size(1))
                                        wave_sem_loss = F.cosine_embedding_loss(
                                            sem_rec_w.reshape(-1, sem_rec_w.size(-1)),
                                            sem_gt_w.reshape(-1, sem_gt_w.size(-1)),
                                            torch.ones(sem_rec_w.size(0) * sem_rec_w.size(1), device=sem_rec_w.device)
                                        )
                                        semantic_loss = semantic_loss + wwave * wave_sem_loss * sem_scale
                                except Exception:
                                    pass

                                # 20→16 蒸馏：对齐 acoustic→semantic 头输出到语义空间
                                try:
                                    distill_feat = decoder_outputs.get('acoustic_semantic_distill', None)
                                    if wdist > 0.0 and distill_feat is not None:
                                        distill_loss = F.cosine_embedding_loss(
                                            distill_feat.reshape(-1, distill_feat.size(-1)),
                                            semantic_features.reshape(-1, semantic_features.size(-1)),
                                            torch.ones(distill_feat.size(0) * distill_feat.size(1), device=distill_feat.device)
                                        )
                                        semantic_loss = semantic_loss + wdist * distill_loss * sem_scale
                                except Exception:
                                    pass
                    except Exception as sem_e:
                        print(f"⚠️ Semantic loss computation failed: {sem_e}")
                        semantic_loss = torch.tensor(0.0, device=device, dtype=feats.dtype)

                except Exception as e:
                    print(f"⚠️ Semantic enhancement failed at step {step}: {e}")
                    # 确保wav不为None，即使语义增强失败
                    if wav is None:
                        print(f"[ERROR] wav is None after semantic enhancement failure, using fallback")
                        try:
                            fallback_out = decoder(z, dec_csi, return_wave=True, target_len=audio.size(-1))
                            _, wav = _normalize_decoder_output(fallback_out)
                        except Exception as fallback_error:
                            print(f"[ERROR] Fallback wave synthesis also failed: {fallback_error}")
                            # 生成零波形作为最后的fallback
                            B, T = feats.shape[:2]
                            target_wav_len = audio.size(-1) if audio is not None else T * 160
                            wav = torch.zeros(B, target_wav_len, device=feats.device, dtype=feats.dtype)
                    # 语义/声学损失在失败时置零，避免后续引用未定义
                    semantic_loss = torch.tensor(0.0, device=device, dtype=feats.dtype)
                    acoustic_loss = torch.tensor(0.0, device=device, dtype=feats.dtype)

            # 最终确保wav不为None
            if wav is None:
                print(f"[CRITICAL ERROR] wav is still None before wave_loss, generating emergency fallback")
                B, T = feats.shape[:2]
                target_wav_len = audio.size(-1) if audio is not None else T * 160
                wav = torch.zeros(B, target_wav_len, device=feats.device, dtype=feats.dtype)

            # Wave loss with optional teacher-forced period mix
            # Robust period estimate: support both decoder-specific estimator and generic mapping
            def _estimate_period_from_feats(fargan_features: torch.Tensor, sr: float = 16000.0) -> torch.Tensor:
                dp = fargan_features[..., 18].float()
                f0_hz = (sr * torch.pow(2.0, dp - 6.5)).clamp(50.0, 400.0)
                return (sr / f0_hz).clamp(32.0, 255.0).round().to(torch.long)

            try:
                period_pred = decoder._estimate_period(feats)
            except Exception:
                period_pred = _estimate_period_from_feats(feats)

            try:
                # 使用与解码器一致的PeriodEstimator计算GT周期，避免映射不一致
                with torch.no_grad():
                    period_gt = decoder.period_estimator(y).to(period_pred.device)
            except Exception:
                period_gt = _estimate_period_from_feats(y).to(period_pred.device)
            tf_ratio = float(getattr(train_one_epoch, '_period_tf_ratio', 0.0))
            # Normalize amplitude: scale predicted wave to target RMS per-sample for stable content loss
            def _squeeze1(x):
                return x.squeeze(1) if x.dim() == 3 and x.size(1) == 1 else x
            wav_b = _squeeze1(wav)
            aud_b = _squeeze1(audio)
            # Per-sample RMS
            eps = 1e-8
            rms_pred = torch.sqrt(wav_b.float().pow(2).mean(dim=-1) + eps)  # [B]
            rms_tgt  = torch.sqrt(aud_b.float().pow(2).mean(dim=-1) + eps)  # [B]
            scale = (rms_tgt / (rms_pred + eps)).clamp(0.25, 4.0).unsqueeze(-1)  # [B,1]
            wav_scaled = wav_b * scale
            # Content loss on RMS-matched signals
            wave_loss_pred, _ = fargan_wave_losses(
                wav_scaled, aud_b, period_pred, device=device, train_weights=wave_train_weights
            )
            if tf_ratio > 0.0:
                # Teacher-forced variant with RMS matching as well
                scale_tf = (rms_tgt / (rms_pred + eps)).clamp(0.25, 4.0).unsqueeze(-1)
                wav_scaled_tf = wav_b * scale_tf
                wave_loss_tf, _ = fargan_wave_losses(
                    wav_scaled_tf, aud_b, period_gt, device=device, train_weights=wave_train_weights
                )
                wave_loss = (1.0 - tf_ratio) * wave_loss_pred + tf_ratio * wave_loss_tf
            else:
                wave_loss = wave_loss_pred
            # Add amplitude (RMS dB) penalty to encourage realistic loudness
            rms_db_pred = 20.0 * torch.log10(rms_pred + eps)
            rms_db_tgt  = 20.0 * torch.log10(rms_tgt + eps)
            amp_pen = torch.mean((rms_db_pred - rms_db_tgt).abs())  # L1 in dB
            amp_pen_w = 0.1
            wave_loss = wave_loss + amp_pen_w * amp_pen
            # Debug: keep raw (pre-weight schedule)
            wave_loss_before_weight = wave_loss.detach().clone()
            wave_weight = StageConfig.scheduled_value(
                stage_cfg.wave_full_schedule, step, stage_cfg.wave_full_weight
            )
            wave_loss = wave_loss * wave_weight

            pred_rms = torch.sqrt(torch.mean(wav.pow(2), dim=-1) + eps)
            pred_rms_db = 20.0 * torch.log10(pred_rms.mean() + eps)

            # Skip rate_loss to avoid NaN issues
            # rate = rate_loss(z, lambda_rate=stage_cfg.lambda_rate)
            balance = balance_loss(logs, stage_cfg.lambda_balance, device=device)
            router = router_consistency_loss(
                logs.get("prob"), logs.get("prob"), lambda_cons=stage_cfg.lambda_cons
            )

            var_t = feats.float().var(dim=1).mean()
            anti_static = (1.0 / (var_t + 1e-3)).clamp(max=1e3)

            # Stage3-style optional stats printing
            try:
                stats_iv = int(getattr(args, 'feat_stats_interval', 20) or 0)
            except Exception:
                stats_iv = 20
            if stats_iv > 0 and (step % stats_iv == 0):
                # Prefer raw 36-d features (pre-fusion) for 36-d stats; fallback to current feats
                try:
                    dh_probe = decoder(z, dec_csi, enable_semantic_output=True, return_wave=False)
                except Exception:
                    dh_probe = None

                try:
                    if isinstance(dh_probe, dict):
                        if 'features_raw' in dh_probe:
                            raw36 = dh_probe['features_raw']
                        elif 'features' in dh_probe:
                            raw36 = dh_probe['features']
                        else:
                            raw36 = feats
                    else:
                        raw36 = feats
                    T = min(raw36.size(1), y.size(1))
                    _print_feature_reconstruction_stats_36(raw36[:, :T, :].float(), y[:, :T, :].float(), step, batch_idx)
                except Exception:
                    pass
                # Try dual-head outputs for 20-d acoustic and 16-d semantic; fallback to unified feats if needed
                dh_out = dh_probe

                # Acoustic 20-d comparison
                try:
                    if isinstance(dh_out, dict) and 'acoustic_features' in dh_out:
                        ac20 = dh_out['acoustic_features']
                    else:
                        ac20 = None
                    if ac20 is not None:
                        T2 = min(ac20.size(1), y.size(1))
                        _print_acoustic20_comparison(ac20[:, :T2, :].float(), y[:, :T2, :20].float(), step, batch_idx)
                except Exception:
                    pass

                # Semantic 16-d stats (with optional target when using extractor)
                try:
                    if isinstance(dh_out, dict) and 'semantic_features' in dh_out:
                        sem16 = dh_out['semantic_features']
                    else:
                        # fallback: last 16 dims from unified feats
                        sem16 = feats[:, :T, 20:36]
                    sem_tgt = None
                    if str(getattr(train_one_epoch, '_semantic_teacher', 'ssl')) == 'extractor':
                        sem_ext = getattr(train_one_epoch, '_semantic_extractor', None)
                        if sem_ext is not None:
                            with torch.no_grad():
                                sem_tgt = sem_ext(audio.detach(), target_frames=sem16.size(1))
                    _print_semantic_stats_16(sem16.float(), sem_tgt.float() if sem_tgt is not None else None, step, batch_idx)
                except Exception:
                    pass

            # 🔧 移除recon损失避免与acoustic_loss/semantic_loss重叠
            # 现在损失结构更清晰：专门的声学损失 + 专门的语义损失 + 波形质量
            # 🎯 调整MoE损失权重，避免与主要损失不平衡
            balance_weight_adjusted = stage_cfg.lambda_balance * 0.5  # 降低MoE平衡损失权重
            router_weight_adjusted = stage_cfg.lambda_cons * 0.3      # 降低路由一致性权重

            # 🔥 联合优化：自适应权重平衡
            # 计算损失平衡权重（避免某个损失过度主导）
            wave_w = alpha_wave
            acoustic_w = max(0.1, min(2.0, wave_loss.item() / max(acoustic_loss.item(), 1e-6)))
            semantic_w = max(0.1, min(1.0, wave_loss.item() / max(semantic_loss.item(), 1e-6)))

            # 联合损失：考虑声学-语义的相互依赖（修复dict键判断）
            cross_modal_loss = 0.0
            try:
                if isinstance(decoder_outputs, dict) \
                   and ('acoustic_features' in decoder_outputs) \
                   and ('semantic_features' in decoder_outputs):
                    # 特征一致性损失：确保声学和语义特征互相支持
                    acoustic_norm = F.normalize(decoder_outputs['acoustic_features'].flatten(1), dim=-1)
                    semantic_norm = F.normalize(decoder_outputs['semantic_features'].flatten(1), dim=-1)
                    cross_consistency = 1.0 - F.cosine_similarity(acoustic_norm, semantic_norm, dim=-1).mean()
                    cross_modal_loss = cross_consistency * 0.1
            except Exception:
                pass

            loss = (
                wave_w * wave_loss           # 波形质量损失
                + acoustic_w * acoustic_loss # 自适应权重的声学损失
                + semantic_w * semantic_loss # 自适应权重的语义损失
                + cross_modal_loss           # 跨模态一致性损失
                + alpha_feat * layered_loss  # 可选分层损失
                + balance_weight_adjusted * balance      # 🔧 调整后的MoE平衡损失
                + router_weight_adjusted * router        # 🔧 调整后的路由一致性损失
                + stage_cfg.anti_static_weight * anti_static  # ✅ 反静态损失
            )
            # Merge decoder-side MoE auxiliary loss (if provided, differentiable)
            try:
                dec_aux_loss = getattr(decoder, '_dec_moe_aux_loss', None)
                if dec_aux_loss is not None and torch.is_tensor(dec_aux_loss) and torch.isfinite(dec_aux_loss):
                    loss = loss + dec_aux_loss
            except Exception:
                pass

            # Check for NaN in loss components before backward
            if torch.isnan(loss).any():
                print(f"⚠️ NaN detected in final loss at step {step}")
                print(f"  recon: {recon.item():.6f}")
                print(f"  wave_loss: {wave_loss.item():.6f}")
                print(f"  acoustic_loss: {acoustic_loss.item():.6f}")
                print(f"  semantic_loss: {semantic_loss.item():.6f}")
                print(f"  cross_modal_loss: {cross_modal_loss:.6f}" if isinstance(cross_modal_loss, torch.Tensor) else f"  cross_modal_loss: {cross_modal_loss:.6f}")
                print(f"  adaptive_weights: wave={wave_w:.2f}, acoustic={acoustic_w:.2f}, semantic={semantic_w:.2f}")
                print(f"  balance: {balance.item():.6f}")
                print(f"  router: {router.item():.6f}")
                print(f"  anti_static: {anti_static.item():.6f}")
                if 'dec_aux_loss' in locals() and dec_aux_loss is not None:
                    print(f"  dec_aux_loss: {dec_aux_loss.item():.6f}")
                # Replace NaN loss with small finite value that maintains gradients
                loss = torch.tensor(0.001, device=loss.device, dtype=loss.dtype, requires_grad=True)

        # Backward pass with mixed precision
        if scaler is not None:
            # Check if loss is finite before scaling
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
            else:
                print(f"⚠️ Non-finite loss detected before scaling at step {step}, skipping backward")
                continue
        else:
            if torch.isfinite(loss):
                loss.backward()
            else:
                print(f"⚠️ Non-finite loss detected at step {step}, skipping backward")
                continue

        # 判别器训练步骤（每隔几步训练一次判别器）
        if use_adversarial_loss and acoustic_adv_loss is not None and disc_optimizer is not None:
            # 每2步训练一次判别器，避免判别器过强
            if step % 2 == 0:
                try:
                    # 获取当前批次的声学特征（detached，避免影响生成器梯度）
                    if float(getattr(train_one_epoch, '_alpha_acoustic', 0.0)) > 0.0:
                        if 'acoustic_features' in locals():
                            # 双头路径
                            pred_acoustic_detached = acoustic_features.detach()
                            target_acoustic_for_disc = acoustic_target
                        else:
                            # 非双头路径
                            pred_acoustic_detached = feats[..., :20].detach()
                            target_acoustic_for_disc = y[..., :20]

                        # 计算判别器损失
                        disc_loss, disc_metrics = acoustic_adv_loss.compute_discriminator_loss(
                            pred_acoustic_detached, target_acoustic_for_disc
                        )

                        # 判别器反向传播
                        disc_optimizer.zero_grad()
                        disc_loss.backward()
                        disc_optimizer.step()

                        # 定期记录判别器指标
                        if step % 40 == 0:
                            print(f"🔍 Discriminator Metrics (Step {step}):")
                            for k, v in disc_metrics.items():
                                print(f"  {k}: {v:.6f}")
                except Exception as e:
                    print(f"⚠️ Discriminator training failed at step {step}: {e}")
        # --- Quick preview audio export (pred + original) ---
        if (
            out_dir is not None and
            isinstance(val_audio_interval, int) and val_audio_interval > 0 and
            (step % val_audio_interval == 0)
        ):
            try:
                snaps = out_dir / 'audio_snaps'
                snaps.mkdir(parents=True, exist_ok=True)
                bidx = 0  # use first sample for previews/diagnostics
                # Prefer an evaluation-mode re-synthesis to avoid training-time noise injection
                pred_wave = None
                try:
                    if hasattr(decoder, 'fargan_core') and decoder.fargan_core is not None:
                        prev_eval = decoder.fargan_core.training
                        decoder.fargan_core.eval()
                        with torch.no_grad():
                            _fe, _wv = decoder(z.detach(), csi_for_enc, return_wave=True, target_len=audio.size(-1))
                        pred_wave = _wv[bidx]
                        # restore training mode
                        decoder.fargan_core.train(prev_eval)
                except Exception:
                    pred_wave = None
                # fallback to training-mode waveform if eval-path unavailable
                pred = pred_wave if pred_wave is not None else wav[bidx]
                orig = audio[bidx]
                if pred.dim() > 1:
                    pred = pred.view(-1)
                if orig.dim() > 1:
                    orig = orig.view(-1)
                max_len = int(max(0.0, float(val_audio_seconds)) * 16000) if val_audio_seconds and val_audio_seconds > 0 else min(pred.numel(), orig.numel())
                L = min(pred.numel(), orig.numel(), max_len)
                pred_np = torch.clamp(pred[:L].detach().cpu(), -1.0, 1.0).numpy()
                # Avoid preview-time de/emphasis during revival to prevent low-frequency hum
                deemph = float(val_audio_deemph) if val_audio_deemph is not None else 0.0
                if revival_active:
                    deemph = 0.0
                if deemph > 0.0:
                    y_prev = 0.0
                    for i in range(pred_np.shape[0]):
                        y_prev = float(pred_np[i]) + deemph * y_prev
                        pred_np[i] = y_prev
                orig_np = torch.clamp(orig[:L].detach().cpu(), -1.0, 1.0).numpy()
                _p_pred = snaps / f'step_{step:06d}_pred.wav'
                _p_orig = snaps / f'step_{step:06d}_orig.wav'
                sf.write(str(_p_pred), pred_np, 16000, subtype='PCM_16')
                sf.write(str(_p_orig), orig_np, 16000, subtype='PCM_16')
                try:
                    safe_print(f"[AB] wrote {_p_pred.resolve()} exists={_p_pred.exists()} size={( _p_pred.stat().st_size if _p_pred.exists() else 0)}")
                    safe_print(f"[AB] wrote {_p_orig.resolve()} exists={_p_orig.exists()} size={( _p_orig.stat().st_size if _p_orig.exists() else 0)}")
                except Exception:
                    pass
                # Extra A/B previews only at revival step 0
                if revival_active and step == 0:
                    try:
                        nb_pre = 2
                        a = audio
                        if a.dim() == 3:
                            a = a.squeeze(1)
                        pre = a[bidx:bidx+1, : nb_pre * 160].contiguous()
                        # 1) AMP+train variant of vocoder (diagnose training-time hiss)
                        try:
                            if hasattr(decoder, 'fargan_core'):
                                prev_mode = decoder.fargan_core.training
                                decoder.fargan_core.train(True)
                            else:
                                prev_mode = None
                            # Estimate period from predicted features
                            with torch.no_grad():
                                per_dbg = decoder._estimate_period(feats.detach())
                            # Use AMP dtype if available
                            amp_dtype = getattr(train_one_epoch, '_amp_dtype', torch.float16)
                            feats_b = feats[bidx:bidx+1].detach()
                            try:
                                from torch.amp import autocast as _ab_autocast
                                with _ab_autocast('cuda', enabled=True, dtype=amp_dtype):
                                    wav_amp = decoder._generate_waveform(feats_b, per_dbg, target_len=audio.size(-1), fargan_pre=pre)
                            except Exception:
                                with autocast(enabled=True, dtype=amp_dtype):
                                    wav_amp = decoder._generate_waveform(feats_b, per_dbg, target_len=audio.size(-1), fargan_pre=pre)
                            wav_amp_np = torch.clamp(wav_amp[bidx].detach().cpu().squeeze().float(), -1.0, 1.0).numpy()
                            _p_amp = snaps / f'step_{step:06d}_pred_amp_train.wav'
                            sf.write(str(_p_amp), wav_amp_np, 16000, subtype='PCM_16')
                            try:
                                safe_print(f"[AB] wrote {_p_amp.resolve()} exists={_p_amp.exists()} size={( _p_amp.stat().st_size if _p_amp.exists() else 0)}")
                            except Exception:
                                pass
                        except Exception:
                            safe_print("[AB] failed to save pred_amp_train.wav")
                        finally:
                            try:
                                if prev_mode is not None and hasattr(decoder, 'fargan_core'):
                                    decoder.fargan_core.train(prev_mode)
                            except Exception:
                                pass
                        # 2) Replace frame_corr with GT (diagnose gating drift)
                        try:
                            feats_g = feats.clone()
                            if feats_g.size(-1) > 19 and y.size(-1) > 19:
                                feats_g[:, :, 19] = y[:, :, 19]
                            feats_g_b = feats_g[bidx:bidx+1].detach()
                            with torch.no_grad():
                                try:
                                    per_g = decoder._estimate_period(feats_g_b)
                                except Exception:
                                    dp = feats_g_b[..., 18].float()
                                    f0_hz = (16000.0 * torch.pow(2.0, dp - 6.5)).clamp(50.0, 400.0)
                                    per_g = (16000.0 / f0_hz).clamp(32.0, 255.0).round().to(torch.long)
                            try:
                                from torch.amp import autocast as _ab_autocast
                                # Force eval mode for vocoder to avoid training-time noise
                                prev_mode2 = None
                                if hasattr(decoder, 'fargan_core'):
                                    prev_mode2 = decoder.fargan_core.training
                                    decoder.fargan_core.eval()
                                with _ab_autocast('cuda', enabled=False):
                                    wav_g = decoder._generate_waveform(feats_g_b.float(), per_g, target_len=audio.size(-1), fargan_pre=pre)
                            except Exception:
                                from torch.amp import autocast as _ab_autocast
                                with _ab_autocast('cuda', enabled=False):
                                    wav_g = decoder._generate_waveform(feats_g_b.float(), per_g, target_len=audio.size(-1), fargan_pre=pre)
                            finally:
                                try:
                                    if prev_mode2 is not None and hasattr(decoder, 'fargan_core'):
                                        decoder.fargan_core.train(prev_mode2)
                                except Exception:
                                    pass
                            wav_g_np = torch.clamp(wav_g[bidx].detach().cpu().squeeze().float(), -1.0, 1.0).numpy()
                            _p_gtc = snaps / f'step_{step:06d}_pred_gtcorr.wav'
                            sf.write(str(_p_gtc), wav_g_np, 16000, subtype='PCM_16')
                            try:
                                safe_print(f"[AB] wrote {_p_gtc.resolve()} exists={_p_gtc.exists()} size={( _p_gtc.stat().st_size if _p_gtc.exists() else 0)}")
                            except Exception:
                                pass
                        except Exception:
                            safe_print("[AB] failed to save pred_gtcorr.wav")
                        # 3) Pure vocoder from GT features (diagnose vocoder weights)
                        try:
                            # Use ground-truth features (full 36-dim) and GT period
                            y_b = y[bidx:bidx+1].detach()
                            # Prefer Stage2 mapping for period if reference vocoder is available
                            per_gt = None
                            if _RefFARGANDecoder is not None:
                                try:
                                    ref_tmp = _build_ref_vocoder_from_stage2(device)
                                    if ref_tmp is not None:
                                        with torch.no_grad():
                                            per_gt = ref_tmp.period_estimator(y_b)
                                except Exception:
                                    per_gt = None
                            if per_gt is None:
                                with torch.no_grad():
                                    per_gt = decoder.period_estimator(y_b)
                            prev_mode3 = None
                            if hasattr(decoder, 'fargan_core'):
                                prev_mode3 = decoder.fargan_core.training
                                decoder.fargan_core.eval()
                            try:
                                from torch.amp import autocast as _ab_autocast
                                with _ab_autocast('cuda', enabled=False):
                                    wav_vc = decoder._generate_waveform(y_b.float(), per_gt, target_len=audio.size(-1), fargan_pre=pre)
                            except Exception:
                                from torch.amp import autocast as _ab_autocast
                                with _ab_autocast('cuda', enabled=False):
                                    wav_vc = decoder._generate_waveform(y_b.float(), per_gt, target_len=audio.size(-1), fargan_pre=pre)
                            wav_vc_np = torch.clamp(wav_vc[bidx].detach().cpu().squeeze().float(), -1.0, 1.0).numpy()
                            _p_gt36 = snaps / f'step_{step:06d}_pred_from_gt36.wav'
                            sf.write(str(_p_gt36), wav_vc_np, 16000, subtype='PCM_16')
                            try:
                                safe_print(f"[AB] wrote {_p_gt36.resolve()} exists={_p_gt36.exists()} size={( _p_gt36.stat().st_size if _p_gt36.exists() else 0)}")
                            except Exception:
                                pass
                        except Exception:
                            safe_print("[AB] failed to save pred_from_gt36.wav")
                        finally:
                            try:
                                if prev_mode3 is not None and hasattr(decoder, 'fargan_core'):
                                    decoder.fargan_core.train(prev_mode3)
                            except Exception:
                                pass
                        # 4) Reference FARGANDecoder using Stage2 ckpt if available (GT36 + GT period)
                        try:
                            if _RefFARGANDecoder is not None:
                                ref_voc = _build_ref_vocoder_from_stage2(device)
                                if ref_voc is None:
                                    # Fallback: clone current core weights
                                    ref_voc = _RefFARGANDecoder().to(device)
                                    try:
                                        ref_voc.fargan_core.load_state_dict(decoder.fargan_core.state_dict(), strict=False)
                                    except Exception:
                                        pass
                                ref_voc.fargan_core.eval()
                                y_ref = y_b.float()
                                per_ref = per_gt.detach()
                            # Ref vocoder expects pre as [B, L]
                            try:
                                from torch.amp import autocast as _ab_autocast
                                with _ab_autocast('cuda', enabled=False):
                                    # Let reference vocoder estimate its own period to ensure mapping consistency
                                    _per, wav_ref = ref_voc(y_ref, target_len=audio.size(-1), pre=pre)
                            except Exception:
                                from torch.amp import autocast as _ab_autocast
                                with _ab_autocast('cuda', enabled=False):
                                    _per, wav_ref = ref_voc(y_ref, target_len=audio.size(-1), pre=pre)
                                wav_ref_np = torch.clamp(wav_ref[bidx].detach().cpu().squeeze().float(), -1.0, 1.0).numpy()
                                _p_ref = snaps / f'step_{step:06d}_pred_ref_vocoder.wav'
                                sf.write(str(_p_ref), wav_ref_np, 16000, subtype='PCM_16')
                                try:
                                    safe_print(f"[AB] wrote {_p_ref.resolve()} exists={_p_ref.exists()} size={( _p_ref.stat().st_size if _p_ref.exists() else 0)}")
                                except Exception:
                                    pass
                        except Exception:
                            safe_print("[AB] failed to save pred_ref_vocoder.wav")
                    except Exception as _e:
                        safe_print(f"[AB] unexpected error during A/B previews: {_e}")
                # Revival-step diagnostics: dump features/period/z for the saved sample (no large I/O beyond step 0)
                try:
                    if revival_active and step == 0:
                        np.save(str(snaps / f'step_{step:06d}_feats36.npy'), feats[bidx].detach().cpu().numpy())
                        # period_pred is computed below for the loss; if not yet available, estimate now
                        try:
                            per_dbg = period_pred[bidx].detach().cpu().numpy()
                        except Exception:
                            per_dbg = decoder._estimate_period(feats)[bidx].detach().cpu().numpy()
                        np.save(str(snaps / f'step_{step:06d}_period.npy'), per_dbg)
                        np.save(str(snaps / f'step_{step:06d}_z.npy'), z[bidx].detach().cpu().numpy())
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
                safe_print(f"🎧 Saved preview audio at step {step} ({L/16000.0:.1f}s)")
            except Exception as e:
                safe_print(f"[WARN] Failed to save preview audio at step {step}: {e}")
        # Gradient clipping + diagnostics
        if scaler is not None:
            # Unscale gradients before clipping for mixed precision
            scaler.unscale_(optimizer)

        # Check for NaN/inf gradients before clipping (expensive: gated by flags)
        do_gcheck = False
        try:
            g_until = int(getattr(train_one_epoch, '_grad_check_until', 0))
            g_intv  = int(getattr(train_one_epoch, '_grad_check_interval', 0))
            do_gcheck = (step < g_until) or (g_intv > 0 and ((batch_idx + 1) % g_intv == 0))
        except Exception:
            pass
        has_nan_grad = False
        if do_gcheck:
            for name, param in list(encoder.named_parameters()) + list(decoder.named_parameters()):
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"⚠️ NaN/Inf gradient detected in {name} at step {step}")
                        has_nan_grad = True

        if has_nan_grad:
            print(f"⚠️ Skipping optimizer step due to NaN/Inf gradients at step {step}")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.update()
            continue

        total_norm = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0
        )
        if not hasattr(train_one_epoch, '_gn_ema'):
            train_one_epoch._gn_ema = 0.0
        with torch.no_grad():
            tn = float(total_norm) if torch.isfinite(total_norm) else 10.0
            beta = 0.98
            train_one_epoch._gn_ema = beta * train_one_epoch._gn_ema + (1.0 - beta) * tn

        # Key gradient diagnostics (FiLM focused)
        if (batch_idx + 1) % max(1, int(log_interval)) == 0:
            film_gn, film_n = _sum_grad_norm(encoder.named_parameters(), include_key='film')
            # Decoder MoE grad norm
            dec_moe_gn = 0.0
            try:
                if hasattr(decoder, 'dec_moe') and decoder.dec_moe is not None:
                    dec_moe_gn, _ = _sum_grad_norm(decoder.dec_moe.named_parameters())
            except Exception:
                pass
            film_info = getattr(encoder, '_last_film_stats', None)
            pre_s = film_info.get('pre_s', 0.0) if isinstance(film_info, dict) else 0.0
            post_s = film_info.get('post_s', 0.0) if isinstance(film_info, dict) else 0.0
            film_pos = getattr(encoder, 'film_position', 'none')
            # Channel summary (batch-level) — only for keys present
            snr_mean = None
            ber_mean = None
            k_db = doppler = tau_ms = coh_fr = lp = blm = None
            try:
                if isinstance(csi_cond, dict) and 'snr_db' in csi_cond:
                    snr_mean = float(csi_cond['snr_db'].detach().mean().item())
                if isinstance(csi_cond, dict) and 'ber' in csi_cond:
                    ber_mean = float(csi_cond['ber'].detach().mean().item())
                if isinstance(csi_cond, dict) and 'k_factor_db' in csi_cond:
                    k_db = float(csi_cond['k_factor_db'].detach().mean().item())
                if isinstance(csi_cond, dict) and 'doppler_norm' in csi_cond:
                    doppler = float(csi_cond['doppler_norm'].detach().mean().item())
                if isinstance(csi_cond, dict) and 'tau_rms_ms' in csi_cond:
                    tau_ms = float(csi_cond['tau_rms_ms'].detach().mean().item())
                if isinstance(csi_cond, dict) and 'coherence_frames' in csi_cond:
                    coh_fr = float(csi_cond['coherence_frames'].detach().mean().item())
                if isinstance(csi_cond, dict) and 'loss_prob' in csi_cond:
                    lp = float(csi_cond['loss_prob'].detach().mean().item())
                if isinstance(csi_cond, dict) and 'burst_len_mean' in csi_cond:
                    blm = float(csi_cond['burst_len_mean'].detach().mean().item())
                # Universal4 proxies
                if isinstance(csi_cond, dict) and 'snr_proxy' in csi_cond:
                    snr_mean = float(csi_cond['snr_proxy'].detach().mean().item())
                if isinstance(csi_cond, dict) and 'time_selectivity' in csi_cond:
                    ts = float(csi_cond['time_selectivity'].detach().mean().item())
                else:
                    ts = None
                if isinstance(csi_cond, dict) and 'freq_selectivity' in csi_cond:
                    fs = float(csi_cond['freq_selectivity'].detach().mean().item())
                else:
                    fs = None
                if isinstance(csi_cond, dict) and 'los_ratio' in csi_cond:
                    los = float(csi_cond['los_ratio'].detach().mean().item())
                else:
                    los = None
            except Exception:
                pass
            # FiLM alpha/beta (a/b) and derived scale/shift
            a_mean = film_info.get('a_mean', 0.0) if isinstance(film_info, dict) else 0.0
            b_mean = film_info.get('b_mean', 0.0) if isinstance(film_info, dict) else 0.0
            sc_mean = film_info.get('scale_mean', 0.0) if isinstance(film_info, dict) else 0.0
            sh_mean = film_info.get('shift_mean', 0.0) if isinstance(film_info, dict) else 0.0
            # EMA tracking for stability
            if not hasattr(train_one_epoch, '_film_ema'):
                train_one_epoch._film_ema = {'a': 0.0, 'b': 0.0}
            beta_ema = 0.98
            train_one_epoch._film_ema['a'] = beta_ema * train_one_epoch._film_ema['a'] + (1.0 - beta_ema) * a_mean
            train_one_epoch._film_ema['b'] = beta_ema * train_one_epoch._film_ema['b'] + (1.0 - beta_ema) * b_mean
            # Assemble dynamic channel postfix
            ch_post = {}
            if snr_mean is not None: ch_post['SNR'] = f"{snr_mean:.1f}dB"
            if k_db is not None:     ch_post['K']   = f"{k_db:.1f}dB"
            if doppler is not None:  ch_post['DPLR']= f"{doppler:.3f}"
            if tau_ms is not None:   ch_post['tau'] = f"{tau_ms:.2f}ms"
            if ber_mean is not None: ch_post['BER'] = f"{ber_mean:.2e}"
            if 'ts' in locals() and ts is not None: ch_post['TS'] = f"{ts:.2f}"
            if 'fs' in locals() and fs is not None: ch_post['FS'] = f"{fs:.2f}"
            if 'los' in locals() and los is not None: ch_post['LOS'] = f"{los:.2f}"

            postfix = {
                'loss': f"{float(loss.item()):.4f}",
                'acou': f"{float(acoustic_loss.item()):.4f}",  # 20维声学损失
                'sem': f"{float(semantic_loss.item()):.4f}",   # 16维语义损失
                'wave': f"{float(wave_loss.item()):.4f}",
                'film_pre': f"{pre_s:.2f}",
                'film_post': f"{post_s:.2f}",
                'film_pos': film_pos,
                'a': f"{a_mean:.3f}",
                'b': f"{b_mean:.3f}",
                'rms': f"{float(pred_rms_db.item()):.1f}dB",
                'g': f"{train_one_epoch._gn_ema:.2e}",
                'g_film': f"{film_gn:.2e}",
                'g_dec': f"{dec_moe_gn:.2e}"
            }
            postfix.update(ch_post)
            # Attach wave debug info
            try:
                postfix['wsrc'] = str(getattr(train_one_epoch, '_feat_source', 'base'))
                postfix['wl_raw'] = f"{float(wave_loss_before_weight.item()):.4f}"
                postfix['wl_wt']  = f"{float(wave_weight):.2f}"
                # amplitude dB difference (batch mean)
                postfix['ampd']  = f"{float((rms_db_pred - rms_db_tgt).mean().item()):+.1f}dB"
            except Exception:
                pass
            progress.set_postfix(postfix)

            # Stage3-style explicit JSCC log line via tqdm.write
            parts = [
                f"step={step}",
                f"loss={float(loss.item()):.4f}",
                f"acou={float(acoustic_loss.item()):.4f}",   # 20维声学损失
                f"sem={float(semantic_loss.item()):.4f}",    # 16维语义损失
                f"wave={float(wave_loss.item()):.4f}",
                f"bal={float((balance_weight_adjusted * balance).item()):.4f}",    # 调整后的MoE平衡损失
                f"rout={float((router_weight_adjusted * router).item()):.4f}",     # 调整后的路由损失
            ]
            # Append present channel keys only
            if snr_mean is not None: parts.append(f"SNR={snr_mean:.1f}dB")
            if ber_mean is not None: parts.append(f"BER={ber_mean:.2e}")
            if k_db is not None:     parts.append(f"K={k_db:.1f}dB")
            if doppler is not None:  parts.append(f"DPLR={doppler:.3f}")
            if tau_ms is not None:   parts.append(f"tau={tau_ms:.2f}ms")
            if coh_fr is not None:   parts.append(f"coh={coh_fr:.1f}f")
            if lp is not None:       parts.append(f"loss_prob={lp:.3f}")
            if blm is not None:      parts.append(f"burst={blm:.2f}")
            if 'ts' in locals() and ts is not None: parts.append(f"TS={ts:.2f}")
            if 'fs' in locals() and fs is not None: parts.append(f"FS={fs:.2f}")
            if 'los' in locals() and los is not None: parts.append(f"LOS={los:.2f}")
            parts.append(f"FiLM(pre={pre_s:.2f}, post={post_s:.2f}, pos={film_pos})")
            parts.append(f"alpha(a)={a_mean:.3f} (ema={train_one_epoch._film_ema['a']:.3f})")
            parts.append(f"beta(b)={b_mean:.3f} (ema={train_one_epoch._film_ema['b']:.3f})")
            parts.append(f"scale={sc_mean:.3f}")
            parts.append(f"shift={sh_mean:.3f}")
            parts.append(f"g_film={film_gn:.2e}")
            parts.append(f"g={train_one_epoch._gn_ema:.2e}")
            # Add wave debug to JSCC line
            try:
                parts.append(f"SRC={getattr(train_one_epoch, '_feat_source', 'base')}")
                parts.append(f"wraw={float(wave_loss_before_weight.item()):.4f}")
                parts.append(f"wwt={float(wave_weight):.2f}")
                parts.append(f"ampd={float((rms_db_pred - rms_db_tgt).mean().item()):+.1f}dB")
            except Exception:
                pass
            # Decoder-side MoE stats (if available)
            try:
                if hasattr(decoder, 'get_dec_moe_stats'):
                    dec_stats = decoder.get_dec_moe_stats() or {}
                    util = dec_stats.get('util')
                    ent = dec_stats.get('entropy')
                    resE = dec_stats.get('residual_energy')
                    # also attach aux losses if present
                    try:
                        aux = getattr(decoder, '_dec_moe_aux', {}) or {}
                        aux_bal = aux.get('balance')
                        aux_ent = aux.get('entropy')
                        if torch.is_tensor(aux_bal):
                            parts.append(f"decBal={float(aux_bal.item()):.3e}")
                        if torch.is_tensor(aux_ent):
                            parts.append(f"decEnt={float(aux_ent.item()):.3f}")
                    except Exception:
                        pass
                    if util:
                        util_s = ','.join(f"{u:.2f}" for u in util)
                        parts.append(f"decU=[{util_s}]")
                    if ent is not None:
                        parts.append(f"decH={ent:.2f}")
                    if resE is not None:
                        parts.append(f"decRes={resE:.3e}")
            except Exception:
                pass
            safe_print("[JSCC] " + " | ".join(parts))

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Optional step checkpoint saving
        if save_every_steps and (save_every_steps > 0) and out_dir is not None:
            try:
                if ((step + 1) % int(save_every_steps)) == 0:
                    # Unwrap DP/DDP modules for clean state_dict keys
                    enc_sd = encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict()
                    dec_sd = decoder.module.state_dict() if hasattr(decoder, 'module') else decoder.state_dict()
                    ckpt = {
                        'epoch': int(epoch_id),
                        'step': int(step + 1),
                        'encoder_state_dict': enc_sd,
                        'decoder_state_dict': dec_sd,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': float(loss.item()),
                    }
                    path = out_dir / f'stage4_step_{step+1:06d}.pth'
                    torch.save(ckpt, path)
                    safe_print(f"💾 Saved step checkpoint: {path}")
            except Exception as _e:
                safe_print(f"[WARN] Failed to save step checkpoint at {step+1}: {_e}")

        bs = x.size(0)
        total += float(loss.item()) * bs
        items += bs

    return {"loss": total / max(1, items), "steps": len(loader)}, float(loss.item())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4: Quantised full optimisation (AETHER+FARGAN)"
    )
    # In combined mode, --features/--pcm are not required
    parser.add_argument("--features", type=str, required=False,
                        help='Features file path (required if not using --combined-data-root)')
    parser.add_argument("--pcm", type=str, required=False,
                        help='Audio PCM file path (required if not using --combined-data-root)')
    parser.add_argument(
        "--stage3-checkpoint",
        type=str,
        default=None,
        help="Optional Stage 3 checkpoint to initialise from.",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints_stage4")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-parallel", action="store_true",
                        help="Use nn.DataParallel to utilize multiple GPUs (simple dual-GPU support)")
    parser.add_argument("--distributed", action="store_true",
                        help="Use DistributedDataParallel for multi-GPU/multi-node training (more efficient than DataParallel)")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="Local rank for distributed training (usually set by torch.distributed.launch)")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use mixed precision training for acceleration")
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"],
                        help="AMP compute dtype: auto→prefer bf16 if supported, else fp16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=800)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-dims", type=int, default=36)
    parser.add_argument("--d-csi", type=int, default=4,
                        help='CSI target dimension for FiLM and decoder refinement (encoder/decoder will use this value)')
    parser.add_argument("--alpha-feat", type=float, default=1.0)
    parser.add_argument("--alpha-wave", type=float, default=1.0)
    # Combined multi-expert dataset (mixed-batch)
    parser.add_argument("--combined-data-root", type=str, default=None,
                        help='Root directory containing expert-mixed datasets for combined loading')
    parser.add_argument("--mix-ratio", type=str, default=None,
                        help='Comma-separated ratios for [harmonic,transient,burst_inpaint,low_snr] in combined mode')
    parser.add_argument("--stride-frames", type=int, default=None,
                        help='Data loader stride in frames (None=auto-adaptive)')
    parser.add_argument("--csi-keys", type=str, default="snr_proxy,time_selectivity,freq_selectivity,los_ratio",
                        help='Comma-separated CSI keys to include (e.g. "snr_db,doppler_norm,k_factor_db,tau_rms_ms").\n'
                             'Available: snr_db, ber, k_factor_db, doppler_norm, tau_rms_ms, coherence_frames, '
                             'loss_prob, burst_len_mean, rate_margin, buffer_level, snr_proxy, time_selectivity, freq_selectivity, los_ratio')
    # MoE topology knobs (optional; can be inferred from Stage3 checkpoint)
    parser.add_argument("--n-experts", type=int, default=None,
                        help="Number of MoE experts. If omitted, tries to infer from Stage3 checkpoint; defaults to 4.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="MoE Top-K routing. If omitted, tries to infer from Stage3 checkpoint; defaults to min(2, n_experts).")
    parser.add_argument("--film", action="store_true", help="Enable FiLM in encoder")
    parser.add_argument("--film-position", type=str, default="post", choices=["pre","post","both"],
                        help="Apply FiLM pre/post MoE (default post for stability)")
    # FiLM tuning knobs (encoder/decoder)
    parser.add_argument("--film-lr-mult", type=float, default=2.0,
                        help='Learning rate multiplier for FiLM parameters')
    parser.add_argument("--film-wd", type=float, default=0.0, help='Weight decay for FiLM parameters')
    parser.add_argument("--film-pre-warmup", type=int, default=500,
                        help='Pre-FiLM warmup steps used in encoder.film schedule')
    parser.add_argument("--film-pre-end", type=float, default=0.15,
                        help='Pre-FiLM target strength (small, default 0.15)')
    parser.add_argument("--film-post-end", type=float, default=0.8,
                        help='Post-FiLM target strength (default 0.8 > 0.6)')
    parser.add_argument("--film-post-warmup", type=int, default=1000,
                        help='Post-FiLM warmup steps (ramp to film-post-end)')
    parser.add_argument("--moe", action="store_true", help="Enable MoE in encoder")
    parser.add_argument("--channel", type=str, default=None, choices=["clean", "awgn", "fading"],
                        help="Channel perturbation type for Stage4 (default from StageConfig)")
    # SNR schedule (high → low)
    parser.add_argument("--snr-hi-db", type=float, default=15.0,
                        help='Initial high-SNR center (dB), e.g., 15')
    parser.add_argument("--snr-lo-db", type=float, default=-5.0,
                        help='Final low-SNR center (dB), e.g., -5')
    parser.add_argument("--snr-ramp-steps", type=int, default=2000,
                        help='Linear ramp steps from hi to lo SNR center')
    parser.add_argument("--snr-window-db", type=float, default=4.0,
                        help='Half-width around SNR center used for uniform sampling (±window/2)')
    # 2FSK modem toggles
    parser.add_argument("--use-2fsk", action="store_true",
                        help="Encode z to bits and 2-FSK modulate through channel (non-differentiable test)")
    parser.add_argument("--fsk-f0", type=float, default=1850.0)
    parser.add_argument("--fsk-f1", type=float, default=1950.0)
    parser.add_argument("--fsk-sym-rate", type=int, default=2000)
    parser.add_argument("--fsk-bits-per-frame", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50, help="Progress bar/log update interval (steps)")
    parser.add_argument("--save-every-steps", type=int, default=0,
                        help="If >0, save a step checkpoint every N global steps")
    # Gradient check frequency (per-parameter NaN scanning is expensive)
    parser.add_argument("--grad-check-until", type=int, default=2,
                        help="Run per-parameter NaN/Inf gradient scan until this global step (0 disables)")
    parser.add_argument("--grad-check-interval", type=int, default=0,
                        help="If >0, also run per-parameter gradient scan every N steps thereafter")
    # Revival: temporarily match Stage3 for the first N steps (disable FiLM/channel/dec_moe)
    parser.add_argument("--revival-steps", type=int, default=200,
                        help='Revival period in steps: disable FiLM, channel and decoder MoE to let Stage3 weights recover')
    parser.add_argument("--channel-start-step", type=int, default=None,
                        help='If set, start applying channel perturbation from this global step (overrides revival)')
    parser.add_argument("--dec-moe-start-step", type=int, default=None,
                        help='If set, enable decoder residual MoE from this global step (overrides revival)')
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a Stage4 checkpoint (.pth). If contains optimizer state, it will be restored.")
    # Preview audio export (disabled by default)
    parser.add_argument("--val-audio-interval", type=int, default=0,
                        help="Every N steps, export pred/orig wav of the first sample (0 to disable)")
    parser.add_argument("--val-audio-seconds", type=float, default=10.0,
                        help="Max seconds per preview clip (default 10s)")
    parser.add_argument("--val-audio-deemph", type=float, default=0.85,
                        help="Preview de-emphasis factor (0 to disable; default 0.85)")
    # Warm-up (freezing) knobs
    parser.add_argument("--freeze-film-steps", type=int, default=0,
                        help="Disable FiLM for first N steps by withholding CSI from encoder/decoder")
    parser.add_argument("--freeze-moe-steps", type=int, default=0,
                        help="Freeze encoder MoE parameters for first N steps")
    parser.add_argument("--freeze-decoder-steps", type=int, default=0,
                        help="Freeze decoder (wave head) parameters for first N steps")
    # Decoder-side MoE knobs
    parser.add_argument("--dec-moe", action='store_true', help='Enable decoder-side residual MoE')
    parser.add_argument("--dec-moe-experts", type=int, default=3)
    parser.add_argument("--dec-moe-topk", type=int, default=2)
    parser.add_argument("--dec-moe-topk-warm-steps", type=int, default=800)
    parser.add_argument("--dec-moe-temp-start", type=float, default=1.5)
    parser.add_argument("--dec-moe-temp-end", type=float, default=0.7)
    parser.add_argument("--dec-moe-temp-steps", type=int, default=1000)
    parser.add_argument("--dec-moe-res-scale-start", type=float, default=0.1)
    parser.add_argument("--dec-moe-res-scale-end", type=float, default=0.2)
    parser.add_argument("--dec-moe-res-scale-steps", type=int, default=1500)
    parser.add_argument("--dec-moe-jitter", type=float, default=5e-3)
    # Decoder MoE routing & regularization
    parser.add_argument("--dec-moe-router-use-csi", action='store_true', default=True,
                        help='Use CSI proxies (snr_proxy/time_selectivity/freq_selectivity/los_ratio) in decoder MoE routing')
    parser.add_argument("--dec-moe-balance-weight", type=float, default=0.2,
                        help='Auxiliary balance loss weight for decoder MoE (encourages uniform expert usage)')
    parser.add_argument("--dec-moe-entropy-weight", type=float, default=0.05,
                        help='Early-phase entropy regularization weight for decoder MoE routing')
    parser.add_argument("--dec-moe-entropy-warm-steps", type=int, default=800,
                        help='Steps to linearly anneal decoder MoE entropy regularization to zero')
    parser.add_argument("--dec-moe-prob-smoothing-eps", type=float, default=0.02,
                        help='Probability smoothing epsilon to avoid zero gradients for tail experts')
    # Decoder FiLM knobs (Refiner)
    parser.add_argument("--dec-film-gain-scale", type=float, default=1.0,
                        help='Decoder-side FiLM gain scale (gamma)')
    parser.add_argument("--dec-film-bias-scale", type=float, default=0.0,
                        help='Decoder-side FiLM bias scale (beta), default 0.0 for pre-gamma-only')
    parser.add_argument("--dec-film-invert", action='store_true',
                        help='Enable inverse FiLM on decoder (encourage CSI removal: gamma=2-alpha, bias negative)')
    parser.add_argument("--dec-film-use-acoustic-priors", action='store_true', default=True,
                        help='Enable frame-level acoustic priors to drive time-varying FiLM (default: enabled)')
    parser.add_argument("--dec-film-ap-blend", type=float, default=0.5,
                        help='Blend ratio for acoustic-priors FiLM vs CSI-FiLM (0..1, default 0.5)')
    # Period smoothing (vocoder robustness)
    parser.add_argument("--period-smooth-ks", type=int, default=3,
                        help='Odd kernel size for period temporal smoothing before vocoder (3=median/mean window; 1 disables)')
    parser.add_argument("--period-smooth-mode", type=str, default='median', choices=['median', 'mean'],
                        help='Smoothing mode for period: median or mean')
    # Wave loss weights (texture tuning)
    parser.add_argument("--wave-w-l1", type=float, default=1.0,
                        help='Wave loss: L1 weight (default 1.0)')
    parser.add_argument("--wave-w-l2", type=float, default=0.2,
                        help='Wave loss: L2 weight (default 0.2)')
    parser.add_argument("--wave-w-energy", type=float, default=0.3,
                        help='Wave loss: energy L1 weight (default 0.3)')
    parser.add_argument("--wave-w-mrstft", type=float, default=1.0,
                        help='Wave loss: multi-resolution STFT L1 weight (increase to enhance texture)')
    parser.add_argument("--wave-w-signal", type=float, default=0.2,
                        help='Wave loss: signal cosine weight (default 0.2)')
    parser.add_argument("--wave-w-continuity", type=float, default=0.02,
                        help='Wave loss: frame-boundary continuity weight (default 0.02)')
    parser.add_argument("--wave-w-pitch", type=float, default=0.05,
                        help='Wave loss: pitch consistency weight (lower slightly to reduce mechanical tone)')
    parser.add_argument("--wave-w-subframe", type=float, default=0.05,
                        help='Wave loss: subframe alignment weight (increase to sharpen micro-textures)')
    # Pitch-guided + period teacher-forcing mix (minimal robust fix)
    parser.add_argument("--feat-pitch-lambda", type=float, default=0.0,
                        help='Extra MSE weight on pitch feature (dim=18) to stabilize F0 tracking (0 disables)')
    parser.add_argument("--period-tf-ratio-start", type=float, default=0.0,
                        help='Initial teacher-forced period mix ratio in wave loss (0..1); linearly anneals to 0')
    parser.add_argument("--period-tf-ratio-steps", type=int, default=1500,
                        help='Anneal steps for period TF ratio; 0 keeps constant ratio')
    parser.add_argument("--period-tf-constant", action='store_true',
                        help='If set, keep TF ratio constant at --period-tf-ratio-start (no anneal)')
    parser.add_argument("--pitch-voiced-thresh", type=float, default=0.10,
                        help='Voiced threshold on frame_corr (dim=19) for pitch-guided loss masking')
    # Decoder MoE light supervised routing (optional)
    parser.add_argument("--dec-moe-trans-supervise", action='store_true',
                        help='Bias a chosen expert when transient feature is high (early steps only)')
    parser.add_argument("--dec-moe-trans-expert-id", type=int, default=1)
    parser.add_argument("--dec-moe-trans-thresh", type=float, default=0.5)
    parser.add_argument("--dec-moe-trans-bias", type=float, default=0.1)
    parser.add_argument("--dec-moe-trans-sup-steps", type=int, default=1000)
    # Per-module LR multipliers for balanced co-training
    parser.add_argument("--moe-lr-mult", type=float, default=1.0,
                        help='LR multiplier for encoder.moe parameters (default 1.0)')
    parser.add_argument("--dec-wave-lr-mult", type=float, default=1.0,
                        help='LR multiplier for decoder wave head (FARGANCore+PeriodEstimator) (default 1.0)')
    parser.add_argument("--dec-moe-lr-mult", type=float, default=1.0,
                        help='LR multiplier for decoder.dec_moe parameters (default 1.0)')
    # 双头解码器和语义损失参数
    # 语义增强解码器控制
    parser.add_argument("--use-dual-head-decoder", action='store_true',
                        help='Use semantic-augmented decoder (DEPRECATED: use --enable-semantic-augmentation)')
    parser.add_argument("--enable-semantic-augmentation", action='store_true',
                        help='Enable semantic augmentation plugin in decoder')
    parser.add_argument("--semantic-enhancement-layers", type=int, default=2,
                        help='Number of layers in semantic enhancement network')
    parser.add_argument("--semantic-dropout", type=float, default=0.1,
                        help='Dropout rate for semantic enhancement network')

    # 对抗损失控制
    parser.add_argument("--use-adversarial-loss", action='store_true', default=True,
                        help='Use adversarial loss for 20D acoustic features (default: True)')
    parser.add_argument("--no-adversarial-loss", dest='use_adversarial_loss', action='store_false',
                        help='Disable adversarial loss, use L1 loss instead')

    # SSL语义监督控制
    parser.add_argument("--ssl-model", type=str, default="hubert-base",
                        choices=["hubert-base", "hubert-large", "wavlm-base", "wavlm-large"],
                        help='SSL model type for semantic supervision')
    parser.add_argument("--alpha-semantic", type=float, default=0.3,
                        help='Weight for semantic alignment loss (match Stage3)')
    parser.add_argument("--alpha-acoustic", type=float, default=1.0,
                        help='Weight for acoustic features loss (20-dim, match Stage3)')
    # 兼容Stage3的参数别名
    parser.add_argument("--alpha-sem", type=float, dest='alpha_semantic',
                        help='Alias for --alpha-semantic (Stage3 compatibility)')
    parser.add_argument("--semantic-loss-type", type=str, default="cosine+infoce",
                        choices=["cosine", "mse", "infonce", "cosine+infoce"],
                        help='Type of semantic alignment loss (no implicit cosine priority)')
    parser.add_argument("--semantic-advanced", type=str, default=None,
                        help='Optional compact overrides for semantic params as k=v list, e.g. "temp=0.8,cw=0.3,iw=0.7,wave=0.3,distill=0.5,layer=9"')
    parser.add_argument("--semantic-teacher", type=str, default="ssl",
                        choices=["ssl", "extractor"],
                        help='Semantic supervision source: ssl (high-dim) or extractor (Stage3 16D)')
    parser.add_argument("--feat-stats-interval", type=int, default=20,
                        help='Interval (steps) to print Stage3-style feature stats (0 disables)')
    # CSI randomisation after given step (strong content dynamics)
    parser.add_argument("--csi-random-start-step", type=int, default=5000,
                        help='Global step after which CSI proxies start strong randomisation (default 5000)')
    parser.add_argument("--csi-random-mode", type=str, default='bimodal', choices=['bimodal', 'uniform'],
                        help='Random mode for CSI proxies after start step: bimodal (good/bad modes) or uniform [0,1]')

    # 语义融合模块控制
    parser.add_argument("--enable-semantic-fusion", action='store_true', default=True,
                        help='Enable semantic fusion module (16D semantic guides 20D acoustic)')
    parser.add_argument("--fusion-type", type=str, default="attention",
                        choices=["attention", "gate", "cross_mlp"],
                        help='Type of semantic fusion mechanism')
    parser.add_argument("--fusion-hidden-dim", type=int, default=64,
                        help='Hidden dimension for semantic fusion module')

    # 运行时语义控制
    parser.add_argument("--disable-semantic-at-runtime", action='store_true',
                        help='Disable semantic processing during forward pass (debugging)')
    parser.add_argument("--semantic-warmup-steps", type=int, default=0,
                        help='Steps to warmup semantic loss (0 = no warmup)')

    # Stage3兼容性参数：输入分流
    parser.add_argument("--split-stream-inputs", action='store_true',
                        help='Enable split-stream input processing (Stage3 compatibility)')

    args = parser.parse_args()

    stage_cfg = get_stage_config("stage4")
    # Optional override via CLI
    if args.channel is not None:
        stage_cfg.apply_channel = (args.channel != "clean")
        stage_cfg.channel_type = args.channel

    # Enable cuDNN autotuner for faster convs (safe with fixed input sizes)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    except Exception:
        pass

    # Initialize distributed training if requested
    distributed_training, rank, world_size, local_rank = setup_distributed()

    # Override local_rank if provided in args (for backward compatibility)
    if args.local_rank != -1:
        local_rank = args.local_rank
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    # Determine device
    if distributed_training:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(
            "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
        )

    out_dir = Path(args.output_dir)
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Wait for main process to create directory
    if distributed_training:
        dist.barrier()

    # JSCC-focused run summary
    safe_print("=" * 60)
    safe_print("JSCC Stage4 Training - Summary")
    safe_print("=" * 60)
    safe_print(f"Device: {device}")
    safe_print(f"Channel: {stage_cfg.channel_type} (apply_channel={stage_cfg.apply_channel})")
    safe_print(f"FiLM: {args.film or stage_cfg.use_film} position={args.film_position}")
    safe_print(f"Split-Stream: {args.split_stream_inputs} (Stage3 compatibility)")
    safe_print(f"SeqLen={args.seq_len} | BatchSize={args.batch_size} | Workers={args.num_workers}")
    safe_print(f"Alpha(feat)={args.alpha_feat} | Alpha(wave)={args.alpha_wave}")

    # --- Data loader: support combined mixed-batch mode (Stage3 style) ---
    if getattr(args, 'combined_data_root', None):
        # Build combined loader from four expert subsets
        train_loader, dataset = create_combined_data_loader(
            data_root=args.combined_data_root,
            sequence_length=args.seq_len,
            batch_size=args.batch_size,
            frame_size=160,
            stride_frames=args.stride_frames,
            energy_selection=True,
            feature_dims=args.feature_dims,
            num_workers=max(1, int(args.num_workers)),
        )
        # Optional custom mix ratio
        if getattr(args, 'mix_ratio', None):
            try:
                ratios = [float(x.strip()) for x in str(args.mix_ratio).split(',')]
                assert len(ratios) == 4
                import numpy as _np
                dataset.mix_ratio = _np.array(ratios, dtype=_np.float64)
                s = dataset.mix_ratio.sum()
                dataset.mix_ratio = dataset.mix_ratio / (s if s > 0 else 1.0)
                dataset.cumprob = _np.cumsum(dataset.mix_ratio)
                print(f"   Combined mix ratio set to: {dataset.mix_ratio.tolist()}")
            except Exception:
                print("⚠️  Invalid --mix-ratio format; expected 'a,b,c,d'. Using defaults.")

        # Update combined data loader for distributed training
        if distributed_training:
            # Create distributed sampler for multi-GPU training
            train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

            # Recreate DataLoader with distributed sampler
            from torch.utils.data import DataLoader
            train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=max(1, int(args.num_workers)),
                pin_memory=True,
                drop_last=True
            )
    else:
        # Fallback: single unified dataset via features/pcm
        if not args.features or not args.pcm:
            raise ValueError("--features and --pcm are required when --combined-data-root is not specified")
        data_root = (
            str(Path(args.features).parent.parent)
            if "lmr_export" in Path(args.features).parts
            else str(Path(args.features).parent)
        )
        train_loader, dataset = create_aether_data_loader(
            data_dir=data_root,
            sequence_length=args.seq_len,
            batch_size=args.batch_size,
            max_samples=None,
            num_workers=max(1, min(4, args.num_workers)),
            energy_selection=True,
            test_mode=False,
            feature_spec_type="fargan",
            features_file=args.features,
            audio_file=args.pcm,
            stride_frames=args.stride_frames,
        )

    # Update data loader for distributed training
    train_sampler = None
    if distributed_training:
        # Create distributed sampler for multi-GPU training
        train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

        # Recreate DataLoader with distributed sampler
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=max(1, int(args.num_workers)),
            pin_memory=True,
            drop_last=True
        )

    # --- Infer MoE topology from Stage3 checkpoint if not provided ---
    def _infer_moe_from_ckpt(ckpt_path: Optional[Path]) -> Tuple[Optional[int], Optional[int]]:
        n_exp: Optional[int] = None
        top_k: Optional[int] = None
        if ckpt_path is None or not ckpt_path.exists():
            return n_exp, top_k
        try:
            payload: Dict[str, Any] = torch.load(str(ckpt_path), map_location="cpu")
        except Exception:
            return n_exp, top_k
        # Prefer explicit args/config if present (epoch/best checkpoints)
        for k in ("args", "config"):
            if isinstance(payload.get(k), dict):
                maybe = payload[k]
                if n_exp is None and isinstance(maybe.get("n_experts"), int):
                    n_exp = maybe.get("n_experts")
                if top_k is None and isinstance(maybe.get("top_k"), int):
                    top_k = maybe.get("top_k")
        # Fallback: count expert indices from encoder_state_dict (works for step checkpoints)
        enc_sd = payload.get("encoder_state_dict")
        if isinstance(enc_sd, dict) and n_exp is None:
            import re
            pat = re.compile(r"^moe\.specialized_moe\.experts\.(\d+)\.")
            idxs = set()
            for key in enc_sd.keys():
                m = pat.match(key)
                if m:
                    try:
                        idxs.add(int(m.group(1)))
                    except Exception:
                        pass
            if idxs:
                n_exp = max(idxs) + 1
        return n_exp, top_k

    inferred_n, inferred_topk = _infer_moe_from_ckpt(Path(args.stage3_checkpoint) if args.stage3_checkpoint else None)
    n_experts = int(args.n_experts) if args.n_experts is not None else (inferred_n if inferred_n is not None else 4)
    top_k = int(args.top_k) if args.top_k is not None else (inferred_topk if inferred_topk is not None else min(2, n_experts))
    # Warn if user-specified MoE topology mismatches Stage3 checkpoint (will reduce weight reuse)
    try:
        if args.n_experts is not None and inferred_n is not None and int(args.n_experts) != int(inferred_n):
            safe_print(f"[WARN] n_experts mismatch: Stage3={inferred_n} vs Stage4 arg={args.n_experts} → many MoE weights won't load")
        if args.top_k is not None and inferred_topk is not None and int(args.top_k) != int(inferred_topk):
            safe_print(f"[WARN] top_k mismatch: Stage3={inferred_topk} vs Stage4 arg={args.top_k}")
    except Exception:
        pass

    print(f"Stage4 MoE topology: n_experts={n_experts}, top_k={top_k}"
          + (" (inferred from Stage3 checkpoint)" if (args.n_experts is None and args.top_k is None and (inferred_n or inferred_topk)) else ""))

    # 现在可以安全地显示MoE信息
    safe_print(f"MoE: {args.moe or stage_cfg.use_moe} experts={n_experts} top_k={top_k}")

    # Resolve CSI keys selection and effective d_csi
    selected_csi_keys = None
    if args.csi_keys:
        selected_csi_keys = [k.strip() for k in str(args.csi_keys).split(',') if k.strip()]
        # If user didn't explicitly change d_csi, auto-match it to number of keys
        if int(args.d_csi) == 10:
            try:
                auto_d = max(1, len(selected_csi_keys))
                safe_print(f"[CSI] Using custom keys ({selected_csi_keys}); auto-set d_csi={auto_d}")
                d_csi_effective = auto_d
            except Exception:
                d_csi_effective = int(args.d_csi)
        else:
            d_csi_effective = int(args.d_csi)
    else:
        d_csi_effective = int(args.d_csi)

    encoder = AETHEREncoder(
        d_in=args.feature_dims,
        d_model=128,
        dz=24,
        d_csi=d_csi_effective,
        use_film=args.film or stage_cfg.use_film,
        film_position=args.film_position,
        use_moe=args.moe or stage_cfg.use_moe,
        use_quantization=True,
        latent_bits=4,
        moe_router_use_csi=False,
        use_semantic_head=True,  # match Stage3 for maximal weight reuse
        n_experts=n_experts,
        top_k=top_k,
        split_stream_inputs=args.split_stream_inputs,  # Stage3兼容性：输入分流
    ).to(device)
    # Keep latent quantization in Stage4
    encoder.quantize_latent = True

    # 创建解码器：根据参数选择语义增强或传统解码器
    enable_semantic = args.enable_semantic_augmentation or args.use_dual_head_decoder  # 向下兼容

    if enable_semantic:
        from models.semantic_augmented_aether_decoder import SemanticAugmentedAETHERDecoder

        decoder_config = {
            "dz": 24,
            "d_out": 36,                          # 保持36维输出
            "d_hidden": 128,
            "d_csi": d_csi_effective,
            "decoder_heads": 2,
            "enable_synth": True,                 # 强制启用FARGAN合成器（波形头）
            "feature_spec_type": "fargan",
            "use_film": True,                     # 保持FiLM
        }

        decoder = SemanticAugmentedAETHERDecoder(
            # AETHERDecoder原有参数
            **decoder_config,
            # 语义增强插件参数
            enable_semantic_augmentation=enable_semantic,
            acoustic_dim=20,
            semantic_dim=16,
            ssl_dim={"hubert-base": 768, "hubert-large": 1024, "wavlm-base": 768, "wavlm-large": 1024}.get(args.ssl_model, 768),
            semantic_enhancement_layers=args.semantic_enhancement_layers,
            semantic_dropout=args.semantic_dropout,
            # 语义融合模块参数
            enable_semantic_fusion=args.enable_semantic_fusion,
            fusion_type=args.fusion_type,
            fusion_hidden_dim=args.fusion_hidden_dim,
        ).to(device)

        safe_print(f"✅ 使用语义增强解码器:")
        safe_print(f"   声学: 20维 → FARGAN波形头合成器 (enable_synth=True)")
        safe_print(f"   语义: 16维 → SSL({args.ssl_model})监督")
        safe_print(f"   增强层数: {args.semantic_enhancement_layers}")
        safe_print(f"   Dropout: {args.semantic_dropout}")

        if args.enable_semantic_fusion:
            safe_print(f"✅ 启用语义融合模块:")
            safe_print(f"   融合类型: {args.fusion_type}")
            safe_print(f"   隐藏维度: {args.fusion_hidden_dim}")
            safe_print(f"   数据流: 36D → 20D+16D → Fusion → Enhanced20D → FARGAN")
        else:
            safe_print("⚠️  语义融合模块已禁用，使用原始20维声学特征")

        # 创建语义监督Teacher
        ssl_teacher = None
        semantic_extractor = None
        if args.semantic_teacher == 'ssl':
            ssl_teacher = load_ssl_model(args.ssl_model, device=device, cache=True)
            ssl_teacher.eval()
            safe_print(f"✅ SSL teacher loaded: {args.ssl_model}")
        else:
            # Stage3风格：16维语义投影教师
            semantic_extractor = create_semantic_extractor(model_name=args.ssl_model, proj_dim=16, device=device)
            semantic_extractor.eval()
            safe_print(f"✅ Semantic extractor initialised: {args.ssl_model}")

        # 运行时语义控制
        if args.disable_semantic_at_runtime:
            decoder.disable_semantic_mode()
            safe_print("⚠️  语义处理已在运行时禁用 (debugging mode)")

    else:
        decoder = AETHERFARGANDecoder(
            dz=24,
            d_out=args.feature_dims,
            d_csi=d_csi_effective,
            enable_synth=True,              # 强制启用FARGAN合成器（波形头）
            feature_spec_type="fargan",
            use_film=True  # Enable by default; revival will temporarily disable it
        ).to(device)
        ssl_teacher = None
        semantic_extractor = None
        safe_print("✅ 使用传统AETHER-FARGAN解码器 (FARGAN波形头已启用)")
        # 非双头路径下，若选择Stage3式Teacher，则初始化语义提取器
        if args.semantic_teacher == 'extractor':
            try:
                semantic_extractor = create_semantic_extractor(model_name=args.ssl_model, proj_dim=16, device=device)
                semantic_extractor.eval()
                safe_print(f"✅ Semantic extractor initialised: {args.ssl_model}")
            except Exception as _e:
                safe_print(f"[WARN] Failed to init semantic extractor: {_e}")
    # Ensure decoder-side calibration starts from identity for safety
    _set_decoder_identity_calib(decoder)
    # Configure decoder-side residual MoE if available
    try:
        if hasattr(decoder, 'dec_moe') and decoder.dec_moe is not None:
            decoder.enable_dec_moe = bool(args.dec_moe)
            decoder.dec_moe.n_experts = int(args.dec_moe_experts)
            decoder.dec_moe.top_k = int(args.dec_moe_topk)
            decoder.dec_moe.topk_warm_steps = int(args.dec_moe_topk_warm_steps)
            decoder.dec_moe.temp_start = float(args.dec_moe_temp_start)
            decoder.dec_moe.temp_end = float(args.dec_moe_temp_end)
            decoder.dec_moe.temp_steps = int(args.dec_moe_temp_steps)
            decoder.dec_moe.res_scale_start = float(args.dec_moe_res_scale_start)
            decoder.dec_moe.res_scale_end = float(args.dec_moe_res_scale_end)
            decoder.dec_moe.res_scale_steps = int(args.dec_moe_res_scale_steps)
            decoder.dec_moe.jitter_std = float(args.dec_moe_jitter)
            # Routing & aux regularization
            decoder.dec_moe.router_use_csi = bool(args.dec_moe_router_use_csi)
            decoder.dec_moe.balance_weight = float(args.dec_moe_balance_weight)
            decoder.dec_moe.entropy_weight = float(args.dec_moe_entropy_weight)
            decoder.dec_moe.entropy_warm_steps = int(args.dec_moe_entropy_warm_steps)
            decoder.dec_moe.prob_smoothing_eps = float(args.dec_moe_prob_smoothing_eps)
            # Light supervision knobs
            decoder.dec_moe.supervise_transient = bool(args.dec_moe_trans_supervise)
            decoder.dec_moe.transient_expert_id = int(args.dec_moe_trans_expert_id)
            decoder.dec_moe.trans_thresh = float(args.dec_moe_trans_thresh)
            decoder.dec_moe.trans_bias = float(args.dec_moe_trans_bias)
            decoder.dec_moe.trans_sup_steps = int(args.dec_moe_trans_sup_steps)
            safe_print(
                f"Decoder MoE: enabled={decoder.enable_dec_moe} | E={decoder.dec_moe.n_experts} "
                f"TopK={decoder.dec_moe.top_k} warm={decoder.dec_moe.topk_warm_steps} "
                f"tau={decoder.dec_moe.temp_start}->{decoder.dec_moe.temp_end} in {decoder.dec_moe.temp_steps} "
                f"res={decoder.dec_moe.res_scale_start}->{decoder.dec_moe.res_scale_end} in {decoder.dec_moe.res_scale_steps} "
                f"jitter={decoder.dec_moe.jitter_std} "
                f"router_csi={decoder.dec_moe.router_use_csi} balW={decoder.dec_moe.balance_weight} entW={decoder.dec_moe.entropy_weight} "
                f"entWarm={decoder.dec_moe.entropy_warm_steps} smooth={decoder.dec_moe.prob_smoothing_eps} "
                f"supT={decoder.dec_moe.supervise_transient} (eid={decoder.dec_moe.transient_expert_id}, thresh={decoder.dec_moe.trans_thresh}, "
                f"bias={decoder.dec_moe.trans_bias}, steps={decoder.dec_moe.trans_sup_steps})"
            )
    except Exception as e:
        safe_print(f"[WARN] Failed to configure decoder MoE: {e}")

    safe_print(f"Model config: d_csi={d_csi_effective}, n_experts={n_experts}, top_k={top_k}, "
               f"quant={encoder.use_quantization}({encoder.latent_bits}b), film={encoder.use_film}, pos={encoder.film_position}")

    # 初始化对抗损失模块（用于20维声学特征）
    use_adversarial_loss = getattr(args, 'use_adversarial_loss', True)
    if use_adversarial_loss:
        acoustic_adv_loss = create_acoustic_adversarial_loss(
            input_dim=20,
            device=device,
            hidden_dim=64,
            num_layers=3,
            recon_weight=1.0,
            adv_weight=0.1,
            use_gradient_penalty=True,
            label_smoothing=0.1
        )
        # 为判别器创建独立的优化器（在lr定义之后创建）
        disc_optimizer = None  # 将在lr定义后再创建
        safe_print(f"✅ Acoustic adversarial loss initialized with {sum(p.numel() for p in acoustic_adv_loss.discriminator.parameters()):,} discriminator parameters")
    else:
        acoustic_adv_loss = None
        disc_optimizer = None

    # Initialize from Stage3 unless resuming Stage4 is requested
    if args.resume:
        safe_print(f"[Resume] Will resume Stage4 from: {args.resume}")
    else:
        _load_stage3_checkpoint(
            encoder,
            decoder,
            Path(args.stage3_checkpoint) if args.stage3_checkpoint else None,
        )
        # After loading Stage3, optionally override vocoder core with Stage2 weights
        try:
            _maybe_load_vocoder_ckpt(decoder)
            # Re-assert identity calibration after any override
            _set_decoder_identity_calib(decoder)
        except Exception:
            pass

    # Configure decoder refiner FiLM scaling (functional symmetry: gain-only by default)
    try:
        if hasattr(decoder, 'refiner'):
            setattr(decoder.refiner, 'film_gain_scale', float(args.dec_film_gain_scale))
            setattr(decoder.refiner, 'film_bias_scale', float(args.dec_film_bias_scale))
            setattr(decoder.refiner, 'film_invert', bool(getattr(args, 'dec_film_invert', False)))
            # 帧级先验FiLM开关与融合比例
            if hasattr(decoder.refiner, 'use_acoustic_priors'):
                decoder.refiner.use_acoustic_priors = bool(getattr(args, 'dec_film_use_acoustic_priors', True))
            if hasattr(decoder.refiner, 'ap_blend'):
                decoder.refiner.ap_blend = float(getattr(args, 'dec_film_ap_blend', 0.5))
            safe_print(f"Decoder Refiner FiLM: gain_scale={getattr(decoder.refiner, 'film_gain_scale', None)} bias_scale={getattr(decoder.refiner, 'film_bias_scale', None)}")
        # Period smoothing knobs on vocoder path
        if hasattr(decoder, 'period_smooth_ks'):
            setattr(decoder, 'period_smooth_ks', int(getattr(args, 'period_smooth_ks', 3) or 3))
        if hasattr(decoder, 'period_smooth_mode'):
            setattr(decoder, 'period_smooth_mode', str(getattr(args, 'period_smooth_mode', 'median') or 'median'))
        safe_print(f"Vocoder period smoothing: ks={getattr(decoder, 'period_smooth_ks', None)} mode={getattr(decoder, 'period_smooth_mode', None)}")
    except Exception as e:
        safe_print(f"[WARN] Failed to configure decoder refiner FiLM scales: {e}")

    # === Stage3语义模块：SemanticFARGANAdapter（初始化但默认bypass） ===
    try:
        # Local import to avoid name resolution issues in some launch contexts
        from models.semantic_fargan_adapter import create_semantic_fargan_adapter
        semantic_adapter = create_semantic_fargan_adapter(
            adapter_type="progressive",
            input_dim=36,
            output_dim=36,
        ).to(device)
        setattr(train_one_epoch, '_semantic_adapter', semantic_adapter)
        safe_print("🔥 Semantic adapter initialised (progressive)")
    except Exception as _e:
        semantic_adapter = None
        safe_print(f"[WARN] Failed to init semantic adapter: {_e}")

    # 将 Teacher 注入到 train_one_epoch 作用域（两者可能其一为 None）
    try:
        setattr(train_one_epoch, '_ssl_teacher', ssl_teacher)
    except Exception:
        pass
    try:
        setattr(train_one_epoch, '_semantic_extractor', semantic_extractor)
    except Exception:
        pass

    lr = stage_cfg.learning_rate

    # 创建判别器优化器（如果启用对抗损失）
    if use_adversarial_loss and acoustic_adv_loss is not None:
        disc_optimizer = optim.Adam(
            acoustic_adv_loss.discriminator.parameters(),
            lr=lr * 0.5,  # 判别器学习率稍低
            betas=(0.5, 0.9)   # GAN训练推荐参数
        )
        safe_print(f"✅ Discriminator optimizer created with lr={lr * 0.5:.2e}")

    # Build param groups with FiLM-specific LR/WD and per-module multipliers
    def split_film_params(module: nn.Module):
        film, nonfilm = [], []
        actual = module.module if hasattr(module, 'module') else module
        for n, p in actual.named_parameters():
            if not p.requires_grad:
                continue
            if 'film' in n.lower():
                film.append(p)
            else:
                nonfilm.append(p)
        return film, nonfilm

    enc_film, enc_nonfilm = split_film_params(encoder)
    dec_film, dec_nonfilm = split_film_params(decoder)

    # Further split non-Film params into MoE/Wave/Dec-MoE groups
    enc_moe, enc_rest = [], []
    enc_actual = encoder.module if hasattr(encoder, 'module') else encoder
    for n, p in enc_actual.named_parameters():
        if not p.requires_grad or ('film' in n.lower()):
            continue
        if '.moe.' in n:
            enc_moe.append(p)
        else:
            enc_rest.append(p)

    dec_wave, dec_moe_params, dec_rest = [], [], []
    dec_actual = decoder.module if hasattr(decoder, 'module') else decoder
    for n, p in dec_actual.named_parameters():
        if not p.requires_grad or ('film' in n.lower()):
            continue
        if n.startswith('fargan_core.') or n.startswith('period_estimator.'):
            dec_wave.append(p)
        elif n.startswith('dec_moe.'):
            dec_moe_params.append(p)
        else:
            dec_rest.append(p)
    param_groups = []
    if enc_rest:
        param_groups.append({"params": enc_rest, "lr": lr, "weight_decay": 1e-6})
    if enc_moe:
        param_groups.append({"params": enc_moe, "lr": lr * float(getattr(args, 'moe_lr_mult', 1.0)), "weight_decay": 1e-6})
    if dec_rest:
        param_groups.append({"params": dec_rest, "lr": lr, "weight_decay": 1e-6})
    if dec_wave:
        param_groups.append({"params": dec_wave, "lr": lr * float(getattr(args, 'dec_wave_lr_mult', 1.0)), "weight_decay": 1e-6})
    if dec_moe_params:
        param_groups.append({"params": dec_moe_params, "lr": lr * float(getattr(args, 'dec_moe_lr_mult', 1.0)), "weight_decay": 1e-6})
    if enc_film:
        param_groups.append({"params": enc_film, "lr": lr * float(args.film_lr_mult), "weight_decay": float(args.film_wd)})
    if dec_film:
        param_groups.append({"params": dec_film, "lr": lr * float(args.film_lr_mult), "weight_decay": float(args.film_wd)})
    # 附加语义适配器参数组（较小学习率、无WD）
    if semantic_adapter is not None:
        try:
            param_groups.append({
                'params': [p for p in semantic_adapter.parameters() if p.requires_grad],
                'lr': lr * 0.1,
                'weight_decay': 0.0
            })
            safe_print("Adapter params added to optimizer (lr x0.1, wd=0)")
        except Exception:
            pass
    optimizer = optim.AdamW(param_groups)
    safe_print(
        f"Param groups -> enc_rest={len(enc_rest)} enc_moe={len(enc_moe)} dec_rest={len(dec_rest)} "
        f"dec_wave={len(dec_wave)} dec_moe={len(dec_moe_params)} | film(enc={len(enc_film)},dec={len(dec_film)}) | "
        f"lr: base={lr:g} film_x={args.film_lr_mult} moe_x={getattr(args,'moe_lr_mult',1.0)} "
        f"decWave_x={getattr(args,'dec_wave_lr_mult',1.0)} decMoe_x={getattr(args,'dec_moe_lr_mult',1.0)} wd_film={args.film_wd}"
    )

    # Multi-GPU setup: support both DataParallel and DistributedDataParallel
    if distributed_training:
        # Use DistributedDataParallel for multi-GPU/multi-node training
        if is_main_process():
            safe_print(f"✅ Using DistributedDataParallel on {world_size} processes")
        encoder = DistributedDataParallel(encoder, device_ids=[local_rank], output_device=local_rank)
        decoder = DistributedDataParallel(decoder, device_ids=[local_rank], output_device=local_rank)
    elif args.data_parallel and device.type == 'cuda' and torch.cuda.device_count() > 1:
        # Use simple DataParallel for single-node, multi-GPU training
        safe_print(f"✅ Using DataParallel on {torch.cuda.device_count()} GPUs")
        encoder = DataParallel(encoder)
        decoder = DataParallel(decoder)

    best = float("inf")
    global_step = 0

    # Resume if requested (after optimizer creation so its state can be restored)
    start_epoch = 1
    if getattr(args, 'resume', None):
        try:
            start_epoch, global_step, best = _resume_stage4(
                encoder, decoder, optimizer, Path(args.resume)
            )
            safe_print(f"[Resume] start_epoch={start_epoch} global_step={global_step} best={best:.6f}")
        except Exception as _e:
            safe_print(f"[WARN] Failed to resume from {args.resume}: {_e}")

    # Prepare FSK modem if requested
    fsk_modem = None
    if args.use_2fsk:
        # Enforce Nyquist
        if args.fsk_f0 >= 0.5 * 16000 or args.fsk_f1 >= 0.5 * 16000:
            raise ValueError("FSK tones must be < Nyquist (8 kHz for 16 kHz sample rate)")
        fsk_modem = TwoFSKModem(sample_rate=16000, f0=args.fsk_f0, f1=args.fsk_f1, sym_rate=args.fsk_sym_rate)

    # Create mixed precision scaler if requested
    scaler = None
    if args.mixed_precision and device.type == 'cuda':
        try:
            # Use new API if available, fallback to old API
            from torch.amp import GradScaler as NewGradScaler
            # Safer settings: lower init_scale, stronger backoff
            scaler = NewGradScaler('cuda', init_scale=2**12, growth_factor=2.0, backoff_factor=0.25, growth_interval=2000)
        except (ImportError, TypeError):
            # Fallback API
            scaler = GradScaler(init_scale=2**12, growth_factor=2.0, backoff_factor=0.25, growth_interval=2000)
        # Decide AMP dtype (prefer bf16 if supported or requested)
        try:
            amp_choice = str(getattr(args, 'amp_dtype', 'auto')).lower()
            if amp_choice == 'bf16':
                amp_dtype = torch.bfloat16
            elif amp_choice == 'fp16':
                amp_dtype = torch.float16
            else:
                bf16_ok = False
                if torch.cuda.is_available():
                    try:
                        bf16_ok = torch.cuda.is_bf16_supported()
                    except Exception:
                        bf16_ok = False
                amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
            setattr(train_one_epoch, '_amp_dtype', amp_dtype)
            print(f"Mixed precision training enabled ({'bf16' if amp_dtype==torch.bfloat16 else 'fp16'}) with safe scaling")
        except Exception:
            setattr(train_one_epoch, '_amp_dtype', torch.float16)
            print(f"Mixed precision training enabled (fp16) with safe scaling")
    elif args.mixed_precision:
        print(f"Warning: Mixed precision requested but device is {device.type}, disabling")

    for epoch in range(start_epoch, args.epochs + 1):
        # Set epoch for distributed sampler (important for shuffling)
        if distributed_training and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Update FiLM schedules (encoder + decoder.refiner if available)
        try:
            # Unconditionally set encoder FiLM schedule knobs (AETHEREncoder uses getattr fallback otherwise)
            setattr(encoder, 'film_pre_warmup', int(args.film_pre_warmup))
            setattr(encoder, 'film_pre_end', float(args.film_pre_end))
            setattr(encoder, 'film_post_end', float(args.film_post_end))
            setattr(encoder, 'film_post_warmup', int(args.film_post_warmup))
            # Decoder refiner may ignore these if unsupported; safe to attach as attributes for future use
            if hasattr(decoder, 'refiner'):
                setattr(decoder.refiner, 'film_pre_warmup', int(args.film_pre_warmup))
                setattr(decoder.refiner, 'film_pre_end', float(args.film_pre_end))
                setattr(decoder.refiner, 'film_post_end', float(args.film_post_end))
                setattr(decoder.refiner, 'film_post_warmup', int(args.film_post_warmup))
            # Print confirmation of FiLM schedules
            try:
                enc_sched = {
                    'pre_end': getattr(encoder, 'film_pre_end', None),
                    'pre_warmup': getattr(encoder, 'film_pre_warmup', None),
                    'post_end': getattr(encoder, 'film_post_end', None),
                    'post_warmup': getattr(encoder, 'film_post_warmup', None)
                }
                dec_sched = {}
                if hasattr(decoder, 'refiner'):
                    for k in ['film_pre_end', 'film_pre_warmup', 'film_post_end', 'film_post_warmup']:
                        if hasattr(decoder.refiner, k):
                            dec_sched[k] = getattr(decoder.refiner, k)
                safe_print(f"[FiLM] Schedules: encoder={enc_sched} decoder_refiner={dec_sched}")
            except Exception:
                pass
        except Exception:
            pass
        # Propagate warm-up configs into the epoch function's static attributes
        setattr(train_one_epoch, 'freeze_film_steps', int(args.freeze_film_steps))
        setattr(train_one_epoch, 'freeze_moe_steps', int(args.freeze_moe_steps))
        setattr(train_one_epoch, 'freeze_decoder_steps', int(args.freeze_decoder_steps))
        # Per-parameter gradient check controls
        setattr(train_one_epoch, '_grad_check_until', int(getattr(args, 'grad_check_until', 2)))
        setattr(train_one_epoch, '_grad_check_interval', int(getattr(args, 'grad_check_interval', 0)))

        # Revival period: disable FiLM/channel/dec_moe for first N steps
        revival = max(0, int(getattr(args, 'revival_steps', 0) or 0))
        # Freeze FiLM by not providing CSI until revival ends
        if revival > 0:
            # If user didn't set freeze_film_steps explicitly, override to revival
            if int(getattr(args, 'freeze_film_steps', 0) or 0) < revival:
                setattr(train_one_epoch, 'freeze_film_steps', revival)
        # Channel/dec_moe start steps
        ch_start = int(getattr(args, 'channel_start_step', revival if revival > 0 else 0) or 0)
        dm_start = int(getattr(args, 'dec_moe_start_step', revival if revival > 0 else 0) or 0)
        setattr(train_one_epoch, '_channel_start_step', ch_start)
        setattr(train_one_epoch, '_dec_moe_start_step', dm_start)

        # 设置双头解码器和语义Teacher相关的标志
        setattr(train_one_epoch, '_use_dual_head', args.use_dual_head_decoder)
        # 训练期可用的语义teacher/提取器/权重设置
        setattr(train_one_epoch, '_ssl_teacher', ssl_teacher if (args.use_dual_head_decoder and 'ssl_teacher' in locals()) else None)
        setattr(train_one_epoch, '_semantic_extractor', semantic_extractor if 'semantic_extractor' in locals() else None)
        setattr(train_one_epoch, '_fusion_reg_weight', getattr(args, 'fusion_reg_weight', 0.1))  # 增加融合正则权重
        setattr(train_one_epoch, '_alpha_acoustic', float(args.alpha_acoustic))
        setattr(train_one_epoch, '_alpha_semantic', float(args.alpha_semantic))
        setattr(train_one_epoch, '_semantic_loss_type', str(args.semantic_loss_type))
        setattr(train_one_epoch, '_semantic_teacher', str(args.semantic_teacher))
        # Semantic loss advanced knobs: keep internal defaults, but allow compact CLI override via --semantic-advanced
        try:
            adv = getattr(args, 'semantic_advanced', None)
            if adv:
                # Parse k=v pairs separated by commas
                kv = {}
                for part in str(adv).split(','):
                    part = part.strip()
                    if not part or '=' not in part:
                        continue
                    k, v = part.split('=', 1)
                    kv[k.strip().lower()] = v.strip()

                def _f(name: str, default: float) -> float:
                    try:
                        return float(kv.get(name, default))
                    except Exception:
                        return default

                def _i(name: str, default: int) -> int:
                    try:
                        return int(float(kv.get(name, default)))
                    except Exception:
                        return default

                setattr(train_one_epoch, '_semantic_temperature', _f('temp', 0.8))
                setattr(train_one_epoch, '_semantic_cosine_weight', _f('cw', 0.3))
                setattr(train_one_epoch, '_semantic_infonce_weight', _f('iw', 0.7))
                setattr(train_one_epoch, '_semantic_wave_weight', _f('wave', 0.3))
                setattr(train_one_epoch, '_semantic_distill_weight', _f('distill', 0.5))
                setattr(train_one_epoch, '_semantic_ssl_layer', _i('layer', 9))
        except Exception:
            pass
        # CSI randomisation knobs
        try:
            setattr(train_one_epoch, '_csi_random_start_step', int(args.csi_random_start_step))
            setattr(train_one_epoch, '_csi_random_mode', str(args.csi_random_mode))
        except Exception:
            setattr(train_one_epoch, '_csi_random_start_step', 5000)
            setattr(train_one_epoch, '_csi_random_mode', 'bimodal')

        # Set pitch-guided lambda, voiced mask threshold and TF ratio BEFORE the epoch begins
        try:
            setattr(train_one_epoch, '_feat_pitch_lambda', float(getattr(args, 'feat_pitch_lambda', 0.0) or 0.0))
            setattr(train_one_epoch, '_pitch_voiced_thresh', float(getattr(args, 'pitch_voiced_thresh', 0.10) or 0.0))
            r0 = float(getattr(args, 'period_tf_ratio_start', 0.0) or 0.0)
            n = int(getattr(args, 'period_tf_ratio_steps', 0) or 0)
            if bool(getattr(args, 'period_tf_constant', False)):
                r = r0
            else:
                if r0 > 0.0:
                    r = r0 if n <= 0 else r0 * max(0.0, 1.0 - float(global_step) / float(n))
                else:
                    r = 0.0
            setattr(train_one_epoch, '_period_tf_ratio', float(max(0.0, min(1.0, r))))
        except Exception:
            pass

        metrics, _ = train_one_epoch(
            encoder,
            decoder,
            train_loader,
            device,
            optimizer,
            args.alpha_feat,
            args.alpha_wave,
            stage_cfg,
            global_step,
            use_2fsk=args.use_2fsk,
            fsk_modem=fsk_modem,
            fsk_bits_per_frame=args.fsk_bits_per_frame,
            log_interval=args.log_interval,
            epoch_id=epoch,
            num_epochs=args.epochs,
            selected_csi_keys=selected_csi_keys,
            save_every_steps=int(getattr(args, 'save_every_steps', 0) or 0),
            out_dir=out_dir,
            # 对抗损失参数传递
            use_adversarial_loss=use_adversarial_loss,
            acoustic_adv_loss=acoustic_adv_loss,
            disc_optimizer=disc_optimizer,
            val_audio_interval=int(getattr(args, 'val_audio_interval', 0) or 0),
            val_audio_seconds=float(getattr(args, 'val_audio_seconds', 10.0) or 0.0),
            val_audio_deemph=float(getattr(args, 'val_audio_deemph', 0.0) or 0.0),
            wave_train_weights={
                'l1': float(args.wave_w_l1),
                'l2': float(args.wave_w_l2),
                'energy': float(args.wave_w_energy),
                'mr_stft': float(args.wave_w_mrstft),
                'signal': float(args.wave_w_signal),
                'continuity': float(args.wave_w_continuity),
                'pitch_consistency': float(args.wave_w_pitch),
                'subframe_alignment': float(args.wave_w_subframe),
            },
            scaler=scaler,
        )
        # Update epoch-local knobs for pitch-guided feature loss and TF ratio anneal
        try:
            setattr(train_one_epoch, '_feat_pitch_lambda', float(args.feat_pitch_lambda))
            r0 = float(args.period_tf_ratio_start)
            n = max(0, int(args.period_tf_ratio_steps))
            if r0 > 0.0:
                # After each epoch pass, recompute base ratio from new global_step
                r = r0 * max(0.0, 1.0 - float(global_step) / float(n if n > 0 else 1)) if n > 0 else r0
            else:
                r = 0.0
            setattr(train_one_epoch, '_period_tf_ratio', float(max(0.0, min(1.0, r))))
        except Exception:
            pass
        # Also propagate SNR schedule knobs to the epoch function for next loop
        setattr(train_one_epoch, 'snr_hi_db', float(args.snr_hi_db))
        setattr(train_one_epoch, 'snr_lo_db', float(args.snr_lo_db))
        setattr(train_one_epoch, 'snr_ramp_steps', int(args.snr_ramp_steps))
        setattr(train_one_epoch, 'snr_window_db', float(args.snr_window_db))
        global_step += metrics.get("steps", 0)

        # Average loss across all processes for distributed training
        epoch_loss = metrics["loss"]
        if distributed_training:
            loss_tensor = torch.tensor(epoch_loss, device=device)
            loss_tensor = reduce_tensor(loss_tensor, world_size)
            epoch_loss = loss_tensor.item()

        if is_main_process():
            safe_print(f"Epoch {epoch}/{args.epochs} loss={epoch_loss:.6f}")

        if epoch_loss < best:
            best = epoch_loss
            # Only save checkpoints on main process
            if is_main_process():
                enc_sd = encoder.module.state_dict() if hasattr(encoder, 'module') else encoder.state_dict()
                dec_sd = decoder.module.state_dict() if hasattr(decoder, 'module') else decoder.state_dict()
                torch.save({
                "encoder_state_dict": enc_sd,
                "decoder_state_dict": dec_sd,
                "epoch": epoch,
                "loss": best,
                }, out_dir / "stage4_best.pth")
                safe_print(f"Saved best Stage4 checkpoint: {out_dir / 'stage4_best.pth'}")

    # Clean up distributed training
    if distributed_training:
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
