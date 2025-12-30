

import torch
import math
def rope(x):
    """
    x: (batch_size, seq_len, dim) — входной тензор
    Возвращает x с применённым Rotary Positional Encoding
    """
    batch_size, seq_len, dim = x.size()
    assert dim % 2 == 0, "Размерность должна быть чётной для разделения на пары"


    pos = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)  # (seq_len, 1)


    dim_half = dim // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, dim_half, 1.0, device=x.device) / dim_half))  # (dim/2)
    freqs = pos * freqs  # (seq_len, dim/2)


    sin = torch.sin(freqs).unsqueeze(0)
    cos = torch.cos(freqs).unsqueeze(0)  # (1, seq_len, dim/2)

    x1 = x[:, :, 0::2]
    x2 = x[:, :, 1::2]


    x_rotated_0 = x1 * cos - x2 * sin
    x_rotated_1 = x2 * cos + x1 * sin

    x_out = torch.stack((x_rotated_0, x_rotated_1), dim=-1)  # (batch, seq_len, dim/2, 2)
    x_out = x_out.flatten(2)  # (batch, seq_len, dim)

    return x_out
