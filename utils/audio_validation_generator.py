#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio validation generator – produce stage-wise reconstructions and comparisons
after training. Each run exports three 10-second segments with original and
reconstructed audio plus visual diagnostics.
"""

import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
import inspect
import matplotlib.pyplot as plt
import librosa

from models.aether_encoder_decoder import AETHEREncoder, AETHERDecoder
from utils.real_data_loader import AETHERRealDataset
from models.utils import build_csi_vec


class AudioValidationGenerator:
    """Utility that generates audio samples and diagnostics for model validation."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.segment_duration = 10.0  # seconds per audio segment
        self.num_segments = 3  # number of segments to export

        # Mel-spectrogram parameters
        self.n_fft = 1024
        self.hop_length = 160  # 10 ms hop
        self.n_mels = 80
        self._mel_eps = 1e-8
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=400,
            hop_length=self.hop_length,
            f_min=50.0,
            f_max=7600.0,
            n_mels=self.n_mels,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            mel_scale="slaney"
        )

    def _ensure_2d_waveform(self, audio_tensor: torch.Tensor, name: str = "audio") -> torch.Tensor:
        """Ensure audio tensor is 2D [B, T] format for consistent processing."""
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # [T] -> [1, T]
        elif audio_tensor.dim() == 3 and audio_tensor.size(1) == 1:
            audio_tensor = audio_tensor.squeeze(1)  # [B, 1, T] -> [B, T]
        elif audio_tensor.dim() > 2:
            # Flatten to 2D if multiple dimensions
            audio_tensor = audio_tensor.view(-1, audio_tensor.size(-1))

        if audio_tensor.dim() != 2:
            raise ValueError(f"{name} should be 2D [B, T], got {audio_tensor.shape}")

        return audio_tensor

    def extract_audio_segments(
        self,
        dataset: AETHERRealDataset,
        target_duration: float = 10.0
    ) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Extract audio segments of the requested duration from the dataset."""
        segments = []
        target_frames = int(target_duration * self.sample_rate / self.hop_length)  # frames for target duration

        for i in range(self.num_segments):
            # choose different positions across the dataset
            start_idx = i * (len(dataset) // self.num_segments)
            batch = dataset[start_idx]  # use standard __getitem__

            # extract features and audio
            features = batch['x'].unsqueeze(0)  # [1, T, 48]
            audio = batch.get('audio', None)  # [audio_length]
            csi_info = batch['csi']

            if audio is not None:
                # ensure audio is a tensor
                if isinstance(audio, tuple):
                    audio = audio[0]
                if not isinstance(audio, torch.Tensor):
                    audio = torch.tensor(audio)

                current_frames = features.shape[1]
                target_T = min(current_frames, target_frames)
                features = features[:, :target_T, :]
                audio_samples = target_T * self.hop_length
                if audio.dim() > 1:
                    audio = audio[0]
                audio = audio[:audio_samples]

                segments.append((features, audio, csi_info))

        return segments

    def compute_log_mel_spectrogram(self, audio: torch.Tensor) -> np.ndarray:
        """Compute a log-Mel spectrogram."""
        if isinstance(audio, torch.Tensor):
            audio_tensor = audio.detach().cpu().to(torch.float32).squeeze()
        else:
            audio_tensor = torch.tensor(np.asarray(audio), dtype=torch.float32).squeeze()

        if audio_tensor.dim() == 0:
            audio_tensor = audio_tensor.unsqueeze(0)

        mel = self.mel_transform(audio_tensor.unsqueeze(0))
        log_mel = (mel + self._mel_eps).log()
        return log_mel.squeeze(0).cpu().numpy()

    def generate_mel_spectrogram_comparison(
        self,
        original_audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        output_path: Path,
        segment_name: str,
        stage_name: str
    ) -> str:
        """Generate a composite log-Mel spectrogram comparison."""
        # compute log-Mel spectrograms for both waveforms
        original_mel = self.compute_log_mel_spectrogram(original_audio)
        reconstructed_mel = self.compute_log_mel_spectrogram(reconstructed_audio)

        # align along time dimension for fair comparison
        time_frames = min(original_mel.shape[1], reconstructed_mel.shape[1])
        original_mel = original_mel[:, :time_frames]
        reconstructed_mel = reconstructed_mel[:, :time_frames]

        # avoid log-scale clipping
        mel_min = float(min(original_mel.min(), reconstructed_mel.min()))
        mel_max = float(max(original_mel.max(), reconstructed_mel.max()))
        if abs(mel_max - mel_min) < 1e-5:
            mel_max = mel_min + 1.0

        # create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # time axis in seconds
        time_axis = np.linspace(0, time_frames * self.hop_length / self.sample_rate, time_frames)

        # Mel frequency axis
        mel_freqs = librosa.mel_frequencies(n_mels=self.n_mels, fmin=50, fmax=7600)
        freq_min, freq_max = float(mel_freqs[0]), float(mel_freqs[-1])

        # original audio spectrogram
        img1 = axes[0, 0].imshow(
            original_mel,
            aspect='auto',
            origin='lower',
            extent=[0, time_axis[-1], freq_min, freq_max],
            cmap='viridis',
            vmin=mel_min,
            vmax=mel_max,
        )
        axes[0, 0].set_title('Original Log-Mel Spectrogram', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Frequency (Hz)')
        plt.colorbar(img1, ax=axes[0, 0], label='Log magnitude')

        # reconstructed audio spectrogram
        img2 = axes[0, 1].imshow(
            reconstructed_mel,
            aspect='auto',
            origin='lower',
            extent=[0, time_axis[-1], freq_min, freq_max],
            cmap='viridis',
            vmin=mel_min,
            vmax=mel_max,
        )
        axes[0, 1].set_title(f'{stage_name} Reconstructed Log-Mel Spectrogram', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        plt.colorbar(img2, ax=axes[0, 1], label='Log magnitude')

        # difference heatmap
        diff_mel = reconstructed_mel - original_mel
        diff_limit = np.percentile(np.abs(diff_mel), 95)
        diff_limit = float(max(diff_limit, 1.0))
        img3 = axes[1, 0].imshow(
            diff_mel,
            aspect='auto',
            origin='lower',
            extent=[0, time_axis[-1], freq_min, freq_max],
            cmap='RdBu_r',
            vmin=-diff_limit,
            vmax=diff_limit,
        )
        axes[1, 0].set_title('Spectral Difference (Reconstructed - Original)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        plt.colorbar(img3, ax=axes[1, 0], label='Difference')

        # mean spectral energy comparison
        original_energy = np.mean(original_mel, axis=1)
        reconstructed_energy = np.mean(reconstructed_mel, axis=1)

        axes[1, 1].plot(mel_freqs, original_energy, label='Original', linewidth=2, color='blue')
        axes[1, 1].plot(mel_freqs, reconstructed_energy, label=f'{stage_name} Reconstructed', linewidth=2, color='red')
        axes[1, 1].set_title('Mean Spectral Energy Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Mean Energy (dB)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log')

        # highlight key frequency regions
        y_min = min(original_energy.min(), reconstructed_energy.min())
        y_max = max(original_energy.max(), reconstructed_energy.max())
        axes[1, 1].axvspan(80, 500, color='yellow', alpha=0.12, label='Fundamental (80-500 Hz)')
        axes[1, 1].text(
            120,
            y_min + 0.05 * (y_max - y_min + 1e-6),
            'Fundamental Region',
            fontsize=10,
            color='orange',
            fontweight='bold'
        )

        # final layout tweaks
        plt.tight_layout()

        # save figure
        png_filename = f"{segment_name}_{stage_name}_mel_comparison.png"
        png_path = output_path / png_filename
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(png_path)

    def extract_f0_trajectory(self, audio: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Extract an F0 trajectory for the given audio."""
        # ensure audio is a 1-D numpy array
        if isinstance(audio, torch.Tensor):
            audio_np = audio.squeeze().cpu().numpy()
        else:
            audio_np = np.array(audio).squeeze()

        # use librosa to estimate F0
        try:
            # STFT-based F0 estimation
            base_kwargs = dict(
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sample_rate,
                hop_length=self.hop_length,
            )
            optional_args = {'threshold': 0.1, 'resolution': 0.1}
            pyin_signature = inspect.signature(librosa.pyin)
            for key, value in optional_args.items():
                if key in pyin_signature.parameters:
                    base_kwargs[key] = value

            f0, voiced_flag, voiced_probs = librosa.pyin(audio_np, **base_kwargs)

            # create a time axis
            time_axis = librosa.frames_to_time(
                np.arange(len(f0)),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )

            return time_axis, f0

        except Exception as e:
            print(f"F0 extraction failed, falling back to autocorrelation: {e}")
            return self._extract_f0_fallback(audio_np)

    def _extract_f0_fallback(self, audio_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback F0 estimation using simple autocorrelation."""
        # frame-level processing
        frame_length = self.n_fft
        hop_length = self.hop_length
        n_frames = 1 + (len(audio_np) - frame_length) // hop_length

        f0_trajectory = []
        time_axis = []

        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end > len(audio_np):
                break

            frame = audio_np[start:end]

            # basic autocorrelation-based F0 detection
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # locate first significant peak
            min_period = int(self.sample_rate / 500)  # 500 Hz max
            max_period = int(self.sample_rate / 80)   # 80 Hz min

            if len(autocorr) > max_period:
                search_range = autocorr[min_period:max_period]
                if len(search_range) > 0:
                    peak_idx = np.argmax(search_range) + min_period
                    f0 = self.sample_rate / peak_idx if peak_idx > 0 else 0
                else:
                    f0 = 0
            else:
                f0 = 0

            f0_trajectory.append(f0)
            time_axis.append(i * hop_length / self.sample_rate)

        return np.array(time_axis), np.array(f0_trajectory)

    def generate_f0_trajectory_comparison(
        self,
        original_audio: torch.Tensor,
        reconstructed_audio: torch.Tensor,
        output_path: Path,
        segment_name: str,
        stage_name: str
    ) -> str:
        """Generate comparison plots for F0 trajectories."""
        # extract F0 trajectories
        orig_time, orig_f0 = self.extract_f0_trajectory(original_audio)
        recon_time, recon_f0 = self.extract_f0_trajectory(reconstructed_audio)

        # create comparison figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # 1) Trajectory overlay
        # replace missing F0 values with NaN so they are skipped in the plot
        orig_f0_clean = np.where((orig_f0 > 0) & (orig_f0 < 1000), orig_f0, np.nan)
        recon_f0_clean = np.where((recon_f0 > 0) & (recon_f0 < 1000), recon_f0, np.nan)

        axes[0].plot(orig_time, orig_f0_clean, label='Original F0', linewidth=2, color='blue', alpha=0.8)
        axes[0].plot(recon_time, recon_f0_clean, label=f'{stage_name} Reconstructed F0', linewidth=2, color='red', alpha=0.8)
        axes[0].set_title('F0 Trajectory Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Fundamental Frequency (Hz)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(80, 500)  # typical speech range

        # topical F0 regions
        axes[0].axhspan(80, 180, alpha=0.1, color='blue', label='Typical Male Range')
        axes[0].axhspan(165, 350, alpha=0.1, color='pink', label='Typical Female Range')

        # 2) Difference analysis
        # align trajectories to compute differences
        min_len = min(len(orig_f0_clean), len(recon_f0_clean))
        if min_len > 0:
            orig_aligned = orig_f0_clean[:min_len]
            recon_aligned = recon_f0_clean[:min_len]
            time_aligned = orig_time[:min_len]

            # only compute differences where both values are valid
            valid_mask = ~(np.isnan(orig_aligned) | np.isnan(recon_aligned))
            if np.any(valid_mask):
                f0_diff = recon_aligned - orig_aligned

                axes[1].plot(time_aligned, f0_diff, color='purple', linewidth=1.5, alpha=0.7)
                axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[1].fill_between(time_aligned, f0_diff, alpha=0.3, color='purple')
                axes[1].set_title('F0 Difference (Reconstructed - Original)', fontsize=14, fontweight='bold')
                axes[1].set_xlabel('Time (s)')
                axes[1].set_ylabel('F0 Difference (Hz)')
                axes[1].grid(True, alpha=0.3)

                axes[1].axhspan(-10, 10, alpha=0.1, color='green', label='Acceptable Error (±10 Hz)')
                axes[1].legend()

        # 3) Statistical summary
        orig_valid = orig_f0_clean[~np.isnan(orig_f0_clean)]
        recon_valid = recon_f0_clean[~np.isnan(recon_f0_clean)]

        if len(orig_valid) > 0 and len(recon_valid) > 0:
            # histogram comparison
            bins = np.linspace(80, 400, 30)
            axes[2].hist(orig_valid, bins=bins, alpha=0.6, label='Original F0',
                        color='blue', density=True)
            axes[2].hist(recon_valid, bins=bins, alpha=0.6, label=f'{stage_name} Reconstructed F0',
                        color='red', density=True)

            # summary statistics
            orig_mean = np.mean(orig_valid)
            recon_mean = np.mean(recon_valid)
            orig_std = np.std(orig_valid)
            recon_std = np.std(recon_valid)

            axes[2].axvline(orig_mean, color='blue', linestyle='--', linewidth=2,
                           label=f'Original Mean: {orig_mean:.1f} Hz')
            axes[2].axvline(recon_mean, color='red', linestyle='--', linewidth=2,
                           label=f'Reconstructed Mean: {recon_mean:.1f} Hz')

            axes[2].set_title('F0 Distribution Comparison', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Fundamental Frequency (Hz)')
            axes[2].set_ylabel('Density')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            stats_text = f'Original: μ={orig_mean:.1f} Hz, σ={orig_std:.1f} Hz\n'
            stats_text += f'Reconstructed: μ={recon_mean:.1f} Hz, σ={recon_std:.1f} Hz\n'
            if len(orig_valid) > 0 and len(recon_valid) > 0:
                # correlation if lengths align
                min_len_valid = min(len(orig_valid), len(recon_valid))
                if min_len_valid > 1:
                    correlation = np.corrcoef(orig_valid[:min_len_valid],
                                            recon_valid[:min_len_valid])[0, 1]
                    stats_text += f'Correlation: {correlation:.3f}'

            axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # tighten layout
        plt.tight_layout()

        # save figure
        png_filename = f"{segment_name}_{stage_name}_f0_comparison.png"
        png_path = output_path / png_filename
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()

        return str(png_path)

    def generate_stage_audio(
        self,
        features: torch.Tensor,
        csi_info: Dict,
        encoder: AETHEREncoder,
        decoder: AETHERDecoder,
        wave_head,  # Any wave head (WaveformDecoderHead, EmbeddedSynthHead, etc.)
        device: torch.device,
        stage_name: str,
        oracle_features: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Generate reconstructed audio for the given stage.

        Args:
            oracle_features: If True, bypass encoder/decoder and feed ground-truth
                features directly into the wave head (teacher-forcing diagnostic).

        Returns:
            Tuple of (waveform, used_silence_fallback flag).
        """
        encoder.eval()
        decoder.eval()
        wave_head.eval()

        with torch.no_grad():
            failed = False

            try:
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                if features.dim() != 3:
                    raise ValueError(f"features must be [B, T, D], got {features.shape}")

                B, _, _ = features.shape
                features = features.to(device)

                # normalize CSI information and match batch dimensions
                csi_dict: Dict[str, torch.Tensor] = {}

                for key, values in csi_info.items():
                    try:
                        if isinstance(values, torch.Tensor):
                            if values.dim() == 0:
                                values = values.view(1)
                            csi_dict[key] = values.clone().detach()
                        elif isinstance(values, (list, tuple)):
                            csi_dict[key] = torch.tensor(values, dtype=torch.float32)
                        else:
                            csi_dict[key] = torch.tensor([values], dtype=torch.float32)
                    except Exception:
                        # fallback defaults
                        default_val = 0.0
                        if key == "snr_db":
                            default_val = 20.0
                        elif key == "ber":
                            default_val = 0.01
                        csi_dict[key] = torch.tensor([default_val], dtype=torch.float32)

                def _ensure_batch_vector(x, default=0.0):
                    if x is None:
                        x = torch.tensor([default], dtype=torch.float32, device=device)
                    elif isinstance(x, (float, int)):
                        x = torch.tensor([float(x)], dtype=torch.float32, device=device)
                    elif isinstance(x, (list, tuple)):
                        x = torch.tensor(list(x), dtype=torch.float32, device=device)
                    elif isinstance(x, torch.Tensor):
                        x = x.to(device=device, dtype=torch.float32)
                    else:
                        x = torch.tensor([default], dtype=torch.float32, device=device)

                    if x.numel() == 0:
                        x = torch.tensor([default], dtype=torch.float32, device=device)

                    x = x.reshape(-1)
                    if x.numel() == 1:
                        x = x.repeat(B)
                    elif x.numel() < B:
                        x = x[:1].repeat(B)
                    else:
                        x = x[:B]
                    return x

                snr = _ensure_batch_vector(csi_dict.get("snr_db"), default=20.0)
                ber = _ensure_batch_vector(csi_dict.get("ber"), default=0.0)

                fading = csi_dict.get("fading_onehot")
                if isinstance(fading, (list, tuple)):
                    fading = torch.tensor(fading, dtype=torch.float32, device=device)
                elif isinstance(fading, torch.Tensor):
                    fading = fading.to(device=device, dtype=torch.float32)
                else:
                    fading = None

                if fading is None:
                    fading = torch.zeros(B, 8, dtype=torch.float32, device=device)
                else:
                    if fading.dim() == 1:
                        if fading.numel() == 8:
                            fading = fading.view(1, 8)
                        else:
                            fading = fading.view(1, -1)
                    if fading.dim() == 2 and fading.size(0) != B:
                        fading = fading[:1, :].repeat(B, 1)
                    elif fading.dim() > 2:
                        fading = fading.view(fading.size(0), -1)
                        fading = fading[:1, :].repeat(B, 1)
                    if fading.size(1) != 8:
                        padded = torch.zeros(B, 8, dtype=torch.float32, device=device)
                        cols = min(fading.size(1), 8)
                        padded[:, :cols] = fading[:B, :cols]
                        fading = padded
                    else:
                        fading = fading[:B]

                # reassemble CSI entries
                normalized_csi = {
                    "snr_db": snr,
                    "ber": ber,
                    "fading_onehot": fading,
                }
                for key, tensor in csi_dict.items():
                    if key in normalized_csi:
                        continue
                    tensor = tensor.to(device=device, dtype=torch.float32)
                    if tensor.dim() == 0:
                        tensor = tensor.view(1)
                    if tensor.dim() == 1:
                        if tensor.size(0) < B:
                            repeats = int(np.ceil(B / tensor.size(0)))
                            tensor = tensor.repeat(repeats)[:B]
                        else:
                            tensor = tensor[:B]
                        tensor = tensor.view(B, 1)
                    else:
                        if tensor.size(0) < B:
                            repeats = int(np.ceil(B / tensor.size(0)))
                            repeat_shape = (repeats,) + (1,) * (tensor.dim() - 1)
                            tensor = tensor.repeat(repeat_shape)[:B]
                        else:
                            tensor = tensor[:B]
                    normalized_csi[key] = tensor

                # encode / decode and synthesize waveform
                if oracle_features:
                    decoded = features
                else:
                    # 让解码器也感知从输入特征提取的声学先验
                    encoded, _ = encoder(features, normalized_csi, inference=True)
                    decoded = decoder(encoded, normalized_csi)
                if decoded.dim() == 2:
                    decoded = decoded.unsqueeze(0)

                target_len = features.shape[1] * self.hop_length
                reconstructed = wave_head(decoded, target_len=target_len)
                reconstructed = reconstructed[..., :target_len]

                return reconstructed.cpu(), failed

            except Exception as e:
                print(f"Warning: failed to generate audio for stage {stage_name}: {e}")
                silence_length = int(self.segment_duration * self.sample_rate)
                failed = True
                return torch.zeros(1, 1, silence_length), failed

    def generate_validation_audio_set(
        self,
        dataset: AETHERRealDataset,
        trained_models: Dict[str, Tuple],  # {stage_name: (encoder, decoder, wave_head)}
        device: torch.device,
        output_dir: str,
        oracle_features: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Generate the full validation audio set.

        Args:
            oracle_features: When True, bypass the encoder/decoder during synthesis so
                the wave head receives ground-truth features. Useful for diagnosing
                whether pitch artifacts originate from the synthesizer.
        """
        print("\nGenerating staged audio validation set")
        print("=" * 50)

        output_path = Path(output_dir) / "audio_validation"
        output_path.mkdir(parents=True, exist_ok=True)

        # extract segments
        print(f"Extracting {self.num_segments} segments of {self.segment_duration}s each...")
        segments = self.extract_audio_segments(dataset, self.segment_duration)

        generated_files = {}

        for seg_idx, (features, original_audio, csi_info) in enumerate(segments):
            print(f"\nProcessing segment {seg_idx + 1}/{len(segments)}")

            segment_files = []

            # save reference audio
            original_file = output_path / f"segment_{seg_idx+1:02d}_original.wav"
            original_audio_np = original_audio.numpy() if isinstance(original_audio, torch.Tensor) else original_audio

            # ensure proper shape and range
            if len(original_audio_np.shape) > 1:
                original_audio_np = original_audio_np.flatten()

            if original_audio_np.max() > 1.0 or original_audio_np.min() < -1.0:
                original_audio_np = original_audio_np / max(abs(original_audio_np.max()), abs(original_audio_np.min()))

            original_audio_np = original_audio_np.astype(np.float32)

            sf.write(str(original_file), original_audio_np, self.sample_rate, subtype='PCM_16')
            segment_files.append(str(original_file))
            print(f"  Saved reference audio: {original_file.name}")

            # generate reconstructions for each stage
            for stage_name, (encoder, decoder, wave_head) in trained_models.items():
                print(f"  Generating {stage_name} reconstruction...")

                try:
                    reconstructed, failed = self.generate_stage_audio(
                        features, csi_info, encoder, decoder, wave_head, device, stage_name,
                        oracle_features=oracle_features
                    )

                    tag = "_FAILED" if failed else ""
                    stage_file = output_path / f"segment_{seg_idx+1:02d}_{stage_name}{tag}.wav"

                    # 使用统一的张量维度处理方法
                    recon_tensor = self._ensure_2d_waveform(reconstructed, f"{stage_name}_reconstructed")
                    # 取第一个batch的音频并转为1D numpy
                    audio_data = recon_tensor[0].detach().cpu().numpy()

                    ref_len = original_audio_np.shape[0]
                    if audio_data.shape[0] > ref_len:
                        audio_data = audio_data[:ref_len]
                    elif audio_data.shape[0] < ref_len:
                        audio_data = np.pad(audio_data, (0, ref_len - audio_data.shape[0]), mode='constant')

                    # loudness match (RMS) for listening; does not affect training
                    try:
                        target_rms = float(np.sqrt(np.mean((original_audio_np.astype(np.float32)) ** 2)) + 1e-8)
                        recon_rms = float(np.sqrt(np.mean((audio_data.astype(np.float32)) ** 2)) + 1e-8)
                        if recon_rms > 0 and target_rms > 0:
                            gain = target_rms / recon_rms
                            gain = float(np.clip(gain, 1.0, 6.0))
                            if recon_rms < 0.25 * target_rms:
                                audio_data = audio_data * gain
                    except Exception:
                        pass

                    audio_data = np.clip(audio_data, -1.0, 1.0)

                    audio_data = audio_data.astype(np.float32)

                    sf.write(str(stage_file), audio_data, self.sample_rate, subtype='PCM_16')
                    segment_files.append(str(stage_file))
                    status = " (silence fallback)" if failed else ""
                    print(f"    Saved {stage_name}: {stage_file.name}{status}")

                    if not failed:
                        print(f"    Creating {stage_name} visualizations...")
                        try:
                            mel_png = self.generate_mel_spectrogram_comparison(
                                original_audio, reconstructed, output_path,
                                f"segment_{seg_idx+1:02d}", stage_name
                            )
                            print(f"      Mel spectrogram comparison: {Path(mel_png).name}")

                            f0_png = self.generate_f0_trajectory_comparison(
                                original_audio, reconstructed, output_path,
                                f"segment_{seg_idx+1:02d}", stage_name
                            )
                            print(f"      F0 trajectory comparison: {Path(f0_png).name}")

                        except Exception as viz_e:
                            print(f"      Warning: failed to generate visualizations: {viz_e}")

                except Exception as e:
                    print(f"    Error: stage {stage_name} failed: {e}")

            generated_files[f"segment_{seg_idx+1}"] = segment_files

        self._generate_playlist(output_path, generated_files)

        print("\nAudio validation set completed")
        print(f"Output directory: {output_path}")
        print(f"Total files: {sum(len(files) for files in generated_files.values())}")

        return generated_files

    def _generate_playlist(self, output_path: Path, generated_files: Dict[str, List[str]]):
        """Create a playlist and visualization index."""
        playlist_file = output_path / "validation_playlist.txt"

        png_files = list(output_path.glob("*.png"))
        png_files.sort()

        with open(playlist_file, 'w', encoding='utf-8') as f:
            f.write("# AETHER staged audio validation playlist\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Format: original audio -> reconstructed audio by stage + visual aids\n\n")

            for segment_name, files in generated_files.items():
                f.write(f"## {segment_name}\n")
                segment_num = segment_name.split('_')[1]

                f.write("### Audio\n")
                for file_path in files:
                    file_name = Path(file_path).name
                    if "original" in file_name:
                        f.write(f"Original: {file_name}\n")
                    else:
                        stage = file_name.split('_')[-1].replace('.wav', '')
                        f.write(f"{stage}: {file_name}\n")

                f.write("\n### Visualizations\n")
                segment_pngs = [png for png in png_files if f"segment_{segment_num}_" in png.name]

                mel_pngs = [png for png in segment_pngs if "mel_comparison" in png.name]
                f0_pngs = [png for png in segment_pngs if "f0_comparison" in png.name]

                if mel_pngs:
                    f.write("Log-Mel spectrogram comparisons:\n")
                    for png in sorted(mel_pngs):
                        stage = png.name.split('_')[2]
                        f.write(f"   - {stage}: {png.name}\n")

                if f0_pngs:
                    f.write("F0 trajectory comparisons:\n")
                    for png in sorted(f0_pngs):
                        stage = png.name.split('_')[2]
                        f.write(f"   - {stage}: {png.name}\n")

                f.write("\n")

            f.write("## Analysis guidelines\n\n")
            f.write("### Log-Mel spectrogram checklist:\n")
            f.write("- Top left: original spectrogram (blue = low energy, yellow = high energy)\n")
            f.write("- Top right: reconstructed spectrogram\n")
            f.write("- Bottom left: spectral difference (blue = deficit, red = excess)\n")
            f.write("- Bottom right: mean energy comparison, highlight fundamental band\n")
            f.write("Common issues: high-frequency roll-off, low-frequency loss, over-smoothed harmonics.\n\n")

            f.write("### F0 trajectory checklist:\n")
            f.write("- Top: F0 traces (blue = original, red = reconstructed)\n")
            f.write("- Middle: difference plot with ±10 Hz acceptable band\n")
            f.write("- Bottom: distribution and correlation statistics\n")
            f.write("Common issues: abrupt F0 jumps, large bias, voiced/unvoiced misclassification.\n\n")

            f.write("### Quick diagnostic flow:\n")
            f.write("1. Listen to the original audio to set the baseline.\n")
            f.write("2. Listen to reconstructions for clarity and naturalness.\n")
            f.write("3. Inspect Mel spectrograms focusing on:\n")
            f.write("   - Low frequencies (0–500 Hz): fundamental and first harmonics\n")
            f.write("   - Mid band (500 Hz–2 kHz): harmonic structure\n")
            f.write("   - High band (2 kHz+): excessive attenuation or noise\n")
            f.write("4. Review F0 trajectories for continuity and accuracy.\n")

        print(f"Playlist file: {playlist_file.name}")
        print(f"Visualization images: {len(png_files)} PNG files")

    def get_audio_comparison_summary(self, generated_files: Dict[str, List[str]]) -> str:
        """Return a textual summary of the generated audio comparisons."""
        total_files = sum(len(files) for files in generated_files.values())
        num_segments = len(generated_files)

        stage_names = set()
        for files in generated_files.values():
            for file_path in files:
                file_name = Path(file_path).name
                if not "original" in file_name:
                    stage = file_name.split('_')[-1].replace('.wav', '')
                    stage_names.add(stage)

        summary = f"""
Audio validation summary:
- Segments: {num_segments}
- Segment duration: {self.segment_duration}s
- Training stages: {len(stage_names)} ({', '.join(sorted(stage_names))})
- Total files: {total_files}
- Comparisons: original audio vs. reconstructions for each stage
- Visuals: log-Mel spectrogram comparisons and F0 trajectory plots

Recommended review workflow:
1. Listening test – assess subjective quality
   - Play the original audio to establish a baseline
   - Listen to reconstructions in stage order and note quality shifts
   - Focus on articulation clarity, background noise, timbre consistency

2. Spectral analysis – objective inspection
   - Check log-Mel spectrograms for frequency-response issues
   - Look for high-frequency roll-off, harmonic loss, over-smoothing
   - Use the difference plot to locate problem bands

3. F0 analysis – prosody verification
   - Inspect trajectory overlays for continuity
   - Review F0 differences and distribution statistics
   - Confirm voiced/unvoiced decisions and pitch stability

Troubleshooting hints:
- Severe high-frequency loss → re-evaluate STFT loss configuration
- Low-frequency dropouts → adjust harmonic loss or gain constraints
- Frequent F0 jumps → revisit pitch smoothing strategies
- Over-smoothed spectra → adjust teacher-forcing or regularization
"""
        return summary


def integrate_audio_validation(
    dataset: AETHERRealDataset,
    trained_models: Dict[str, Tuple],
    device: torch.device,
    output_dir: str,
    oracle_features: bool = False
) -> str:
    """
    Entry point used by the training pipeline to run audio validation.

    Args:
        oracle_features: Forward ground-truth features to the wave head for all
            stages, skipping encoder/decoder. Helpful for isolating synthesizer
            issues without changing the training pipeline.
    """
    generator = AudioValidationGenerator()

    generated_files = generator.generate_validation_audio_set(
        dataset, trained_models, device, output_dir, oracle_features=oracle_features
    )

    summary = generator.get_audio_comparison_summary(generated_files)

    return summary


def export_validation_audio(
    *,
    stage_name: str,
    y_hat_feats: torch.Tensor,
    y_orig_feats: torch.Tensor,
    wave_head,
    original_audio: Optional[torch.Tensor],
    output_dir: Union[Path, str],
    csi_dict: Optional[Dict[str, torch.Tensor]] = None,
    sample_rate: int = 16000,
) -> None:
    """Export quick diagnostic audio for the current stage.

    Saves three wav files (predicted, teacher-forced, reference) so that training can
    inspect what the synth is producing mid-run without running the full validation
    sweep.
    """
    output_root = Path(output_dir) / "audio_validation" / "stage_exports"
    output_root.mkdir(parents=True, exist_ok=True)

    device = y_hat_feats.device
    hop = 160

    def _slice_batch(t: torch.Tensor) -> torch.Tensor:
        if t is None:
            return t
        if t.dim() == 0:
            return t.view(1)
        if t.size(0) > 1:
            return t[:1]
        return t

    # Prepare CSI slice for single-sample synthesis
    csi_single = None
    if csi_dict is not None:
        csi_single = {}
        for key, value in csi_dict.items():
            if isinstance(value, torch.Tensor):
                csi_single[key] = _slice_batch(value).detach().to(device)
            else:
                csi_single[key] = value

    prev_mode = wave_head.training
    wave_head.eval()
    try:
        with torch.no_grad():
            pred_feats = _slice_batch(y_hat_feats).to(device)
            ref_feats = _slice_batch(y_orig_feats).to(device)

            target_len = pred_feats.size(1) * hop

            pred_audio = wave_head(pred_feats, target_len=target_len, csi_dict=csi_single)
            ref_audio = wave_head(ref_feats, target_len=target_len, csi_dict=csi_single)

            def _to_wave(x: torch.Tensor) -> np.ndarray:
                if x.dim() == 3:
                    x = x[0]
                if x.dim() == 2:
                    x = x[0]
                return torch.clamp(x.detach().cpu(), -1.0, 1.0).numpy()

            pred_np = _to_wave(pred_audio)
            ref_np = _to_wave(ref_audio)

            sf.write(str(output_root / f"{stage_name}_pred.wav"), pred_np, sample_rate, subtype="PCM_16")
            sf.write(str(output_root / f"{stage_name}_teacher.wav"), ref_np, sample_rate, subtype="PCM_16")

            if original_audio is not None:
                orig = _slice_batch(original_audio).to(device)
                orig_np = _to_wave(orig)
                sf.write(str(output_root / f"{stage_name}_ref.wav"), orig_np, sample_rate, subtype="PCM_16")
    finally:
        wave_head.train(prev_mode)
