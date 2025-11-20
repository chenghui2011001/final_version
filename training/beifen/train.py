# -*- coding: utf-8 -*-
"""
Reference training loop wiring together the AETHER encoder and decoder.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from ..models.aether_encoder_decoder import AETHERDecoder, AETHEREncoder
    from ..models.waveform_decoder import MRSTFTLoss, WaveformDecoderHead, save_preview
    from ..training.config import TrainConfig
    from ..training.losses import balance_loss, l1_stft_loss, rate_loss, router_consistency_loss
    from ..training.mask import make_time_mask
except ImportError:  # pragma: no cover - support running as a script
    import sys

    SCRIPT_ROOT = Path(__file__).resolve().parents[1]
    if str(SCRIPT_ROOT) not in sys.path:
        sys.path.append(str(SCRIPT_ROOT))
    PROJECT_ROOT = SCRIPT_ROOT.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from models.aether_encoder_decoder import AETHERDecoder, AETHEREncoder  # type: ignore
    from models.waveform_decoder import MRSTFTLoss, WaveformDecoderHead, save_preview  # type: ignore
    from training.config import TrainConfig  # type: ignore
    from training.losses import balance_loss, l1_stft_loss, rate_loss, router_consistency_loss  # type: ignore
    from training.mask import make_time_mask  # type: ignore


def build_models(cfg: TrainConfig) -> Tuple[AETHEREncoder, AETHERDecoder]:
    encoder = AETHEREncoder(
        d_in=48,
        d_model=cfg.d_model,
        dz=cfg.dz,
        gla_depth=cfg.gla_depth,
        n_heads=cfg.n_heads,
        d_csi=cfg.d_csi,
        dropout=cfg.dropout,
        use_film=cfg.use_film,
        use_moe=cfg.use_moe,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        latent_bits=cfg.latent_bits,
        frame_rate_hz=cfg.frame_rate_hz,
        quantize_latent=cfg.quantize_latent,
    )
    decoder = AETHERDecoder(
        dz=cfg.dz,
        d_out=48,
        d_hidden=cfg.d_model,
        d_csi=cfg.d_csi,
        decoder_heads=max(1, cfg.n_heads),
    )
    return encoder, decoder


def apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x * mask.unsqueeze(-1)


class InsightLogger:
    def __init__(self, out_dir: Path, buffer_limit: int = 256) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "insights.jsonl"
        self.buffer: List[Dict[str, object]] = []
        self.buffer_limit = buffer_limit

    def add(self, stage: str, step: int, payload: Dict[str, torch.Tensor]) -> None:
        if not payload:
            return
        batch = 0
        tensors = {}
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                tensors[key] = value.detach().cpu()
                if tensors[key].ndim > 0:
                    batch = max(batch, tensors[key].shape[0])

        for idx in range(batch or 1):
            record: Dict[str, object] = {"step": step, "stage": stage}
            for key, value in tensors.items():
                if value.ndim == 0:
                    record[key] = float(value.item())
                elif value.ndim == 1:
                    if idx < value.size(0):
                        record[key] = float(value[idx].item())
                else:
                    if idx < value.size(0):
                        record[key] = value[idx].tolist()
            if "ribbon_energy" in record and "thread_energy" in record:
                denom = record["thread_energy"] + 1e-6
                record["band_energy_ratio"] = float(record["ribbon_energy"] / denom)
            self.buffer.append(record)

        if len(self.buffer) >= self.buffer_limit:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        with self.log_path.open("a", encoding="utf-8") as fp:
            for record in self.buffer:
                fp.write(json.dumps(record) + "\n")
        self.buffer.clear()


def save_checkpoint(
    out_dir: Path,
    encoder: AETHEREncoder,
    decoder: AETHERDecoder,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    stage: str,
) -> None:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "stage": stage,
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(cfg),
    }
    torch.save(payload, ckpt_dir / f"checkpoint_{step:06d}.pt")


def load_pcm_audio(
    path: Path,
    dtype: str = "int16",
    channels: int = 1,
) -> np.ndarray:
    """
    Load a mono PCM file into a float32 numpy array normalised to [-1, 1].
    """
    if dtype not in {"int16", "float32"}:
        raise ValueError("audio dtype must be 'int16' or 'float32'")
    np_dtype = np.int16 if dtype == "int16" else np.float32
    raw = np.fromfile(path, dtype=np_dtype)
    if channels != 1:
        if raw.size % channels != 0:
            raise ValueError("Audio data is not divisible by the requested channel count.")
        raw = raw.reshape(-1, channels).mean(axis=1)
    if dtype == "int16":
        audio = raw.astype(np.float32) / 32768.0
    else:
        audio = raw.astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def soft_channel_latent(z: torch.Tensor, csi: Dict[str, torch.Tensor], mode: str = "awgn") -> torch.Tensor:
    """
    Apply a lightweight soft channel corruption to latent sequences based on CSI metadata.
    """
    snr = csi.get("snr_db")
    if snr is None:
        return z

    snr_lin = (10.0 ** (snr / 10.0)).view(-1, 1, 1).to(z.device, z.dtype)
    z_std = z.detach().float().std(dim=(1, 2), keepdim=True).clamp_min_(1e-4).to(z.dtype)
    noise_std = z_std / snr_lin.sqrt()
    noise = torch.randn_like(z) * noise_std

    if mode == "awgn":
        return z + noise
    if mode == "awgn+rayleigh":
        fading = csi.get("fading_onehot")
        scale = 0.9
        if fading is not None:
            scale = 0.7 + 0.3 * fading.argmax(dim=-1, keepdim=True).float() / max(fading.size(-1) - 1, 1)
            scale = scale.view(-1, 1, 1).to(z.dtype)
        return scale * z + noise
    return z


def forward_pass(
    batch: Dict[str, torch.Tensor],
    encoder: AETHEREncoder,
    decoder: AETHERDecoder,
    cfg: TrainConfig,
    wave_head: Optional[WaveformDecoderHead],
    wave_loss: Optional[MRSTFTLoss],
    use_audio_loss: bool,
    apply_soft_channel: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    x = batch["x"]
    y = batch.get("y")
    csi = batch["csi"]
    target_audio = batch.get("audio")

    mask = make_time_mask(x.size(0), x.size(1), cfg.mask_ratio, cfg.burst_ratio, csi)
    x_masked = apply_mask(x, mask)

    encoder.active_stage = cfg.stage
    z_quant, logs = encoder(x_masked, csi, attn_mask=mask)

    if apply_soft_channel:
        z_noisy = soft_channel_latent(z_quant, csi, mode="awgn")
    else:
        z_noisy = z_quant

    z_cont = logs.pop("latent_continuous", z_quant)
    y_hat = decoder(z_noisy, csi, attn_mask=mask)

    # Estimate effective bitrate via entropy of latent codes
    bits_per_frame_est = None
    kbps_proxy = None
    if cfg.quantize_latent and encoder.latent_bits > 0:
        levels = 2 ** int(encoder.latent_bits)
        idx = ((z_quant.detach() + 1.0) * 0.5 * (levels - 1)).round().clamp_(0, levels - 1).long()
        flat_cpu = idx.view(-1, idx.size(-1)).cpu()
        entropies = []
        for d in range(flat_cpu.size(1)):
            hist = torch.bincount(flat_cpu[:, d], minlength=levels).float()
            prob = hist / (hist.sum() + 1e-12)
            entropy = -(prob * (prob + 1e-12).log2()).sum()
            entropies.append(entropy)
        if entropies:
            bits_per_frame_est = torch.stack(entropies).sum().to(z_quant.device)
            kbps_proxy = bits_per_frame_est * cfg.frame_rate_hz / 1000.0

    waveform_hat: Optional[torch.Tensor] = None
    if use_audio_loss and target_audio is not None and wave_head is not None and wave_loss is not None:
        waveform_hat = wave_head(y_hat, target_len=target_audio.size(-1))
        recon = wave_loss(waveform_hat, target_audio)
    else:
        if y is None:
            raise RuntimeError("Feature targets missing while audio loss is disabled.")
        # Only compute loss on unmasked positions (mask=1 means keep, mask=0 means drop)
        keep_positions = mask.unsqueeze(-1)
        recon = l1_stft_loss(y_hat * keep_positions, y * keep_positions, cfg.lambda_stft)
    latent_reg = rate_loss(z_cont, cfg.lambda_rate)
    # Enable MoE balance loss in all stages, not just Stage C
    balance = balance_loss(logs, cfg.lambda_balance, device=x.device)

    consistency = torch.zeros((), device=x.device, dtype=x.dtype)
    # Compute consistency loss when MoE is active (now enabled in all stages)
    if cfg.use_moe and "prob" in logs:
        mask_alt = make_time_mask(x.size(0), x.size(1), cfg.mask_ratio, cfg.burst_ratio, csi)
        x_alt = apply_mask(x, mask_alt)
        with torch.no_grad():
            _, logs_alt = encoder(x_alt, csi, attn_mask=mask_alt)
        consistency = router_consistency_loss(
            logs.get("prob"),
            logs_alt.get("prob") if logs_alt else None,
            cfg.lambda_cons,
        ).to(x.device)

    loss = recon + latent_reg + balance + consistency

    losses = {
        "loss": loss,
        "recon": recon,
        "rate": latent_reg,
        "balance": balance,
        "consistency": consistency,
    }

    insights = {
        "snr_db": logs.get("snr_db"),
        "alpha_mean": logs.get("alpha_mean"),
        "beta_mean": logs.get("beta_mean"),
        "moe_prob": logs.get("prob"),
        "ribbon_energy": logs.get("ribbon_energy"),
        "thread_energy": logs.get("thread_energy"),
        "bits_per_frame_nominal": logs.get("bits_per_frame_nominal"),
        "kbps_nominal": logs.get("kbps_nominal"),
    }

    if waveform_hat is not None:
        insights["waveform_rms"] = waveform_hat.pow(2).mean(dim=-1).sqrt()

    if bits_per_frame_est is not None:
        insights["bits_per_frame_est"] = bits_per_frame_est.detach()
        insights["kbps"] = kbps_proxy.detach()
    else:
        insights.setdefault("bits_per_frame_est", torch.tensor(0.0, device=x.device, dtype=x.dtype))
        if "kbps" not in insights:
            insights["kbps"] = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    insights["lambda_rate"] = torch.tensor(cfg.lambda_rate, device=x.device, dtype=x.dtype)

    return loss, losses, insights, waveform_hat


def train_step(
    batch: Dict[str, torch.Tensor],
    encoder: AETHEREncoder,
    decoder: AETHERDecoder,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer,
    wave_head: Optional[WaveformDecoderHead],
    wave_loss: Optional[MRSTFTLoss],
    use_audio_loss: bool,
    apply_soft_channel: bool,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    use_amp = scaler is not None

    optimizer.zero_grad(set_to_none=True)

    if use_amp:
        with torch.amp.autocast("cuda"):
            loss, losses, insights, waveform_hat = forward_pass(
                batch, encoder, decoder, cfg, wave_head, wave_loss, use_audio_loss, apply_soft_channel
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        params_to_clip = list(encoder.parameters()) + list(decoder.parameters())
        if use_audio_loss and wave_head is not None:
            params_to_clip += list(wave_head.parameters())
        torch.nn.utils.clip_grad_norm_(params_to_clip, cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss, losses, insights, waveform_hat = forward_pass(
            batch, encoder, decoder, cfg, wave_head, wave_loss, use_audio_loss, apply_soft_channel
        )
        loss.backward()
        params_to_clip = list(encoder.parameters()) + list(decoder.parameters())
        if use_audio_loss and wave_head is not None:
            params_to_clip += list(wave_head.parameters())
        torch.nn.utils.clip_grad_norm_(params_to_clip, cfg.grad_clip)
        optimizer.step()

    metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
    return metrics, insights, waveform_hat


class RandomFeatureDataset(Dataset):
    """
    Tiny synthetic dataset that mirrors the expected dataloader contract.
    """

    def __init__(self, length: int = 256, seq_len: int = 120, freeze_csi: bool = False) -> None:
        super().__init__()
        self.length = length
        self.seq_len = seq_len
        self.freeze_csi = freeze_csi

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch.manual_seed(idx)
        t = torch.linspace(0, 1, steps=self.seq_len)

        base_pitch = torch.rand(1).item() * 180 + 60
        harmonics = torch.tensor([1.0, 2.0, 4.0, 8.0])
        phase = t.unsqueeze(-1) * harmonics * base_pitch * 2 * math.pi
        sin_part = torch.sin(phase)
        cos_part = torch.cos(phase)
        f0_pe = torch.cat([sin_part, cos_part], dim=-1)

        voiced_mask = (torch.rand(self.seq_len) > 0.25).float()
        silence = (torch.rand(self.seq_len) < 0.1).float()
        vuv = torch.stack(
            [
                voiced_mask,
                1.0 - voiced_mask,
                1.0 - silence,
                torch.clamp(voiced_mask + torch.randn(self.seq_len) * 0.05, 0.0, 1.0),
            ],
            dim=-1,
        )

        energy_scale = torch.rand(1).item() * 0.6 + 0.7
        mel = torch.randn(self.seq_len, 36) * energy_scale + 0.1

        x = torch.cat([mel, f0_pe, vuv], dim=-1)
        y = x + 0.02 * torch.randn_like(x)

        snr = torch.empty(1).uniform_(-5.0, 15.0)
        if self.freeze_csi:
            fading = torch.zeros(1, 8)
            fading[0, 0] = 1.0
            ber = torch.zeros(1)
        else:
            fading = torch.nn.functional.one_hot(torch.randint(0, 8, (1,)), num_classes=8).float()
            ber = torch.rand(1) * 0.05
        csi = {"snr_db": snr, "fading_onehot": fading, "ber": ber}
        return {"x": x, "y": y, "csi": csi}


class FeatureSequenceDataset(Dataset):
    """Dataset reading real features_48.f32 exports and aligned audio segments."""

    def __init__(
        self,
        feature_path: Path,
        seq_len: int = 120,
        jitter: float = 0.02,
        freeze_csi: bool = False,
        audio_path: Optional[Path] = None,
        audio_dtype: str = "int16",
        sample_rate: int = 16000,
        hop_length: int = 160,
        stride_frames: Optional[int] = None,
        random_offset: bool = False,
        limit_seqs: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        self.feature_path = feature_path
        self.seq_len = int(seq_len)
        self.jitter = float(jitter)
        self.freeze_csi = bool(freeze_csi)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)

        total_floats = feature_path.stat().st_size // 4
        if total_floats % 48 != 0:
            raise ValueError("Expected features_48.f32 to have dimension divisible by 48")
        self.total_frames = total_floats // 48
        if self.total_frames < self.seq_len:
            raise ValueError("Insufficient frames for the requested sequence length")

        self.features = np.memmap(feature_path, dtype=np.float32, mode="r", shape=(self.total_frames, 48))
        self.audio: Optional[np.ndarray] = None

        if audio_path is not None:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio = load_pcm_audio(audio_path, dtype=audio_dtype, channels=1)
            self.audio = audio.astype(np.float32)

        # Sequence start indices with stride/random offset
        self.stride = int(stride_frames) if stride_frames and stride_frames > 0 else self.seq_len
        offset = np.random.randint(0, self.stride) if random_offset else 0
        max_start = max(0, self.total_frames - self.seq_len)
        starts = list(range(offset, max_start + 1, self.stride))
        if limit_seqs is not None:
            starts = starts[: int(limit_seqs)]
        self.starts = np.asarray(starts, dtype=np.int64)
        if self.starts.size == 0:
            raise ValueError("No sequences produced; consider adjusting seq-len or stride.")
        self.num_sequences = int(self.starts.size)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        end = start + self.seq_len
        segment = np.array(self.features[start:end], copy=True)
        x = torch.from_numpy(segment)
        # Apply feature normalization to prevent loss explosion
        x = torch.clamp(x, -10.0, 10.0)  # Clamp extreme values

        if self.jitter > 0:
            y = x + self.jitter * torch.randn_like(x)
        else:
            y = x.clone()

        snr = torch.empty(1, dtype=torch.float32).uniform_(-5.0, 15.0)
        if self.freeze_csi:
            fading = torch.zeros(1, 8, dtype=torch.float32)
            fading[0, 0] = 1.0
            ber = torch.zeros(1, dtype=torch.float32)
        else:
            fading = torch.nn.functional.one_hot(torch.randint(0, 8, (1,)), num_classes=8).float()
            ber = torch.rand(1, dtype=torch.float32).mul_(0.05)
        csi = {"snr_db": snr, "fading_onehot": fading, "ber": ber}
        sample: Dict[str, torch.Tensor] = {"x": x, "y": y, "csi": csi}

        if self.audio is not None:
            start_sample = start * self.hop_length
            desired = self.seq_len * self.hop_length
            end_sample = start_sample + desired
            a = self.audio[start_sample:end_sample]
            if a.size < desired:
                pad = np.zeros((desired - a.size,), dtype=np.float32)
                a = np.concatenate([a, pad], axis=0)
            sample["audio"] = torch.from_numpy(a.copy())
        return sample


def collate_fn(batch: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    xs, ys, snrs, fadings, bers, audios = [], [], [], [], [], []
    for sample in batch:
        xs.append(sample["x"])
        ys.append(sample["y"])
        snrs.append(sample["csi"]["snr_db"])
        fadings.append(sample["csi"]["fading_onehot"])
        if "ber" in sample["csi"]:
            bers.append(sample["csi"]["ber"])
        if "audio" in sample:
            audios.append(sample["audio"])
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    snr = torch.cat(snrs, dim=0)
    fading = torch.cat(fadings, dim=0)
    csi = {"snr_db": snr, "fading_onehot": fading}
    if bers:
        csi["ber"] = torch.cat(bers, dim=0)
    batch_dict: Dict[str, torch.Tensor] = {"x": x, "y": y, "csi": csi}
    if audios:
        audio_tensor = torch.stack(audios, dim=0).unsqueeze(1)
        batch_dict["audio"] = audio_tensor
    return batch_dict


def derive_schedule(cfg: TrainConfig, override_steps: int | None) -> List[int]:
    base_schedule = list(cfg.stage_schedule)
    total = sum(base_schedule)
    if override_steps is None:
        return base_schedule
    if override_steps == total:
        return base_schedule
    if override_steps < total:
        ratio = override_steps / total
        scaled = [max(1, int(round(ratio * s))) for s in base_schedule]
        diff = override_steps - sum(scaled)
        scaled[-1] += diff
        return scaled
    # override_steps > total, extend final stage
    scaled = base_schedule
    scaled[-1] += override_steps - total
    return scaled


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AETHER reference model.")
    parser.add_argument("--steps", type=int, default=None, help="Number of optimisation steps to run.")
    parser.add_argument("--features-path", type=Path, default=None, help="Path to features_48.f32 export")
    parser.add_argument("--audio-path", type=Path, default=None, help="Aligned PCM audio used as reconstruction target")
    parser.add_argument("--audio-dtype", choices=["int16", "float32"], default="int16", help="PCM dtype for audio-path")
    parser.add_argument("--seq-len", type=int, default=120, help="Sequence length (frames) per training sample")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for audio targets")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true", help="Enable torch.amp autocast")
    parser.add_argument("--target-kbps", type=float, default=1.0, help="Target average kbps for latent bitrate control")
    parser.add_argument("--kbps-tol", type=float, default=0.2, help="Tolerance band around target kbps")
    parser.add_argument("--lambda-rate-init", type=float, default=1e-3, help="Initial lambda for rate penalty")
    parser.add_argument("--rho", type=float, default=1e-3, help="Dual ascent step size for kbps control")
    parser.add_argument("--lambda-update-interval", type=int, default=50, help="Steps between rate lambda updates")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train (overrides --steps)")
    parser.add_argument("--stage-steps", type=str, default=None, help="Explicit step counts for stages A,B,C (e.g. '1200,800,1500')")
    parser.add_argument("--stride-frames", type=int, default=None, help="Frame stride between sequences (<seq-len enables overlap)")
    parser.add_argument("--random-offset", action="store_true", help="Randomise starting frame offset within stride")
    parser.add_argument("--limit-seqs", type=int, default=None, help="Limit number of sequences used for training")
    parser.add_argument("--jitter", type=float, default=0.02, help="Noise level applied to feature targets")
    parser.add_argument("--csi-dim", type=int, default=None, help="Override TrainConfig.d_csi")
    parser.add_argument("--freeze-csi", action="store_true", help="Disable FiLM/MoE adaptation while still sampling CSI metadata")
    parser.add_argument("--disable-soft-channel", action="store_true", help="Skip latent-domain channel noise injection")
    parser.add_argument("--preview-seconds", type=float, default=5.0, help="Length of preview audio clips")
    parser.add_argument("--preview-every", type=int, default=500, help="Steps between preview waveform dumps")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.target_kbps = args.target_kbps
    cfg.kbps_tolerance = args.kbps_tol
    cfg.lambda_rate = args.lambda_rate_init
    cfg.lambda_rate_rho = args.rho
    cfg.lambda_update_interval = args.lambda_update_interval

    if args.csi_dim is not None:
        cfg.d_csi = args.csi_dim

    freeze_csi = bool(args.freeze_csi)
    if freeze_csi and args.csi_dim is None:
        cfg.d_csi = 10  # Match actual CSI dimensions from dataset (1 + 8 + 1 = 10)
    if freeze_csi:
        cfg.use_film = False
        cfg.use_moe = False

    encoder, decoder = build_models(cfg)
    decoder_out_dim = getattr(decoder, "d_out", 48)
    wave_head = WaveformDecoderHead(in_dim=decoder_out_dim)
    wave_loss = MRSTFTLoss()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    encoder.to(device)
    decoder.to(device)
    wave_head.to(device)
    wave_loss.to(device)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_root = Path(cfg.run_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    logger = InsightLogger(logs_dir, buffer_limit=cfg.insight_buffer_limit)

    if args.audio_path is not None and not args.audio_path.exists():
        raise FileNotFoundError(f"Audio path not found: {args.audio_path}")

    if args.features_path is not None:
        dataset: Dataset = FeatureSequenceDataset(
            args.features_path,
            seq_len=args.seq_len,
            jitter=args.jitter,
            freeze_csi=freeze_csi,
            audio_path=args.audio_path,
            audio_dtype=args.audio_dtype,
            sample_rate=args.sample_rate,
            hop_length=160,
            stride_frames=args.stride_frames,
            random_offset=args.random_offset,
            limit_seqs=args.limit_seqs,
        )
    else:
        dataset = RandomFeatureDataset(seq_len=args.seq_len, freeze_csi=freeze_csi)

    use_audio_loss = isinstance(dataset, FeatureSequenceDataset) and getattr(dataset, "audio", None) is not None
    base_soft_channel = not bool(args.disable_soft_channel)

    if args.stage_steps:
        parts = [max(1, int(p)) for p in args.stage_steps.split(",")]
        if len(parts) != 3:
            raise ValueError("--stage-steps must specify three comma-separated integers")
        schedule = parts
    elif args.epochs is not None:
        steps_per_epoch = max(1, math.ceil(len(dataset) / (cfg.batch_size or 1)))
        total_steps = int(args.epochs * steps_per_epoch)
        schedule = derive_schedule(cfg, total_steps)
    else:
        schedule = derive_schedule(cfg, args.steps)

    if freeze_csi or not cfg.use_moe:
        total = sum(schedule)
        schedule = [total, 0, 0]
    elif not cfg.use_film:
        schedule = [schedule[0] + schedule[1], 0, schedule[2]]

    opt_params = list(encoder.parameters()) + list(decoder.parameters())
    if use_audio_loss:
        opt_params += list(wave_head.parameters())
    optimizer = torch.optim.AdamW(opt_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    preview_dir = run_root / "previews"
    if use_audio_loss:
        preview_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loader_iter = iter(loader)

    encoder.train()
    decoder.train()
    wave_head.train()

    def move_to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        csi = {k: v.to(device, non_blocking=True) for k, v in batch["csi"].items()}
        moved: Dict[str, torch.Tensor] = {"x": x, "y": y, "csi": csi}
        audio = batch.get("audio")
        if audio is not None:
            moved["audio"] = audio.to(device, non_blocking=True)
        return moved

    stages = ["A", "B", "C"]
    global_step = 0
    for stage_name, stage_steps in zip(stages, schedule):
        cfg.stage = stage_name
        encoder.active_stage = stage_name
        stage_step = 0  # Counter for current stage
        for _ in range(stage_steps):
            global_step += 1
            stage_step += 1

            # FiLM progressive activation for Stage B
            if cfg.stage == "B" and cfg.use_film:
                w = max(1, cfg.film_warmup_steps)
                prog = min(1.0, stage_step / float(w))
                film_ratio = cfg.film_start_ratio + (1.0 - cfg.film_start_ratio) * prog
                # 前半段把 beta 缩小，后半段恢复
                beta_scale = cfg.film_beta_scale + (1.0 - cfg.film_beta_scale) * prog
                encoder.set_film_activation(film_ratio, beta_scale)
            else:
                encoder.set_film_activation(1.0, 1.0)
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            batch = move_to_device(batch)

            metrics, insights, waveform_hat = train_step(
                batch,
                encoder,
                decoder,
                cfg,
                optimizer,
                wave_head,
                wave_loss if use_audio_loss else None,
                use_audio_loss,
                (base_soft_channel and stage_name != "A"),
                scaler=scaler if use_amp else None,
            )

            kbps_tensor = insights.get("kbps")
            kbps_val: Optional[float] = None
            if isinstance(kbps_tensor, torch.Tensor):
                kbps_val = float(kbps_tensor.mean().item())

            if global_step % cfg.log_interval == 0:
                kbps_str = f" kbps={kbps_val:.2f}" if kbps_val is not None else ""

                # FiLM activation status for Stage B
                film_str = ""
                if cfg.stage == "B" and cfg.use_film:
                    film_str = f" film={encoder.film_ratio:.2f} β={encoder.film_beta_scale:.2f}"

                # Enhanced monitoring with stability checks
                instability_warning = ""
                if metrics['recon'] > 1000.0:
                    instability_warning += " [HIGH_RECON_LOSS!]"
                if metrics['loss'] != metrics['loss']:  # NaN check
                    instability_warning += " [NAN_LOSS!]"
                if kbps_val is not None and kbps_val > cfg.target_kbps * 3:
                    instability_warning += " [HIGH_BITRATE!]"

                print(
                    f"[{global_step:05d}][Stage {stage_name}] "
                    f"loss={metrics['loss']:.4f} rec={metrics['recon']:.4f} "
                    f"rate={metrics['rate']:.4f} bal={metrics['balance']:.4f} "
                    f"cons={metrics['consistency']:.4f}{kbps_str}{film_str} λ_rate={cfg.lambda_rate:.6f}{instability_warning}"
                )

            logger.add(stage_name, global_step, insights)

            # Early stopping for extreme instability
            if metrics['recon'] > 10000.0 or metrics['loss'] != metrics['loss']:
                print(f"CRITICAL: Training instability detected at step {global_step}. Stopping.")
                print(f"Reconstruction loss: {metrics['recon']:.2f}, Total loss: {metrics['loss']:.2f}")
                break

            if kbps_val is not None:
                prev = getattr(cfg, "kbps_ema", kbps_val)
                cfg.kbps_ema = 0.9 * prev + 0.1 * kbps_val
                if global_step % cfg.lambda_update_interval == 0:
                    err = (cfg.kbps_ema - cfg.target_kbps) / max(cfg.kbps_tolerance, 1e-6)
                    cfg.lambda_rate = float(max(1e-6, cfg.lambda_rate + cfg.lambda_rate_rho * err))

            if use_audio_loss and waveform_hat is not None and global_step % max(1, args.preview_every) == 0:
                preview_len = int(args.preview_seconds * args.sample_rate)
                preview_wave = waveform_hat[0, 0].detach().cpu()
                if preview_len < preview_wave.numel():
                    preview_wave = preview_wave[:preview_len]
                snr_tensor = insights.get("snr_db")
                kbps_preview = kbps_val if kbps_val is not None else cfg.target_kbps
                snr_val = float(snr_tensor.mean().item()) if isinstance(snr_tensor, torch.Tensor) else 0.0
                preview_path = preview_dir / f"preview_step{global_step:05d}_snr{snr_val:+04.1f}_kbps{kbps_preview:04.2f}.wav"
                save_preview(preview_wave, args.sample_rate, preview_path)

            if global_step % cfg.flush_interval == 0:
                logger.flush()

            if global_step % cfg.checkpoint_interval == 0:
                save_checkpoint(run_root, encoder, decoder, optimizer, cfg, global_step, stage_name)

    logger.flush()
    save_checkpoint(run_root, encoder, decoder, optimizer, cfg, global_step, cfg.stage)


if __name__ == "__main__":
    main()
