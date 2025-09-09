from typing import Optional, Union

import torch
import numpy as np


class F0Predictor(object):
    def __init__(
        self,
        hop_length=512,
        f0_min=50,
        f0_max=1100,
        sampling_rate=44100,
        device: Optional[str] = None,
    ):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

    def compute_f0(
        self,
        wav: np.ndarray,
        p_len: Optional[int] = None,
        filter_radius: Optional[Union[int, float]] = None,
    ): ...

    def _interpolate_f0(self, f0: np.ndarray):
        """
        对F0进行插值处理
        """

        data = np.reshape(f0, (f0.size, 1))

        vuv_vector = np.zeros((data.size, 1), dtype=np.float32)
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0

        ip_data = data

        frame_number = data.size
        last_value = 0.0
        for i in range(frame_number):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]  # 这里可能存在一个没有必要的拷贝
                last_value = data[i]

        return ip_data[:, 0], vuv_vector[:, 0]

    def _resize_f0(self, x: np.ndarray, target_len: int):
        source = np.array(x)
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * target_len, len(source)) / target_len,
            np.arange(0, len(source)),
            source,
        )
        res = np.nan_to_num(target)
        return res
