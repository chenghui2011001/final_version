# -*- coding: utf-8 -*-
"""
零参数包装器：把 decoder 的帧特征直接喂给其内部 OLA 合成器。
这样 progressive_train 里仍然以 'wave_head(...)' 的形式调用。
"""

import torch
import torch.nn as nn


class EmbeddedSynthHead(nn.Module):
    """
    纯"特征→波形"包装器：接收解码后的 [B,T,48] 特征，输出波形 [B,1,T_audio]
    规范化接口：wave_head(decoded_feats, target_len) -> waveform
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self._is_exciter = False  # 兼容现有分支判断

        # 确保 decoder 启用了内嵌合成
        assert hasattr(decoder, "synth") and decoder.synth is not None, "Decoder 未启用合成分支"

    def forward(self, decoded_feats: torch.Tensor, target_len: int = None, csi_dict=None):
        """
        接收解码后的特征，直接合成波形
        Args:
            decoded_feats: [B, T, 48] 解码器输出的特征
            target_len: 目标波形长度
            csi_dict: 兼容参数（未使用）
        Returns:
            waveform: [B, T_audio] 合成波形（压缩维度以匹配训练预期）
        """
        wav = self.decoder.synth(decoded_feats, target_len=target_len)  # [B, 1, T_audio]
        return wav.squeeze(1)  # [B, T_audio]