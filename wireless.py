import numpy as np
from numpy.linalg import norm

def channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts):
# Tính suy hao đường truyền (Path Loss) dưới dạng dB
    path_loss_db = path_loss_ref + path_loss_exp * np.log10(distances_RU_UE / 1000)
    path_loss_linear = 10**(-path_loss_db / 10)
    # Tạo ma trận channel gain
    gain = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))

    # Tính độ lợi kênh cho mỗi cặp RU-UE
    for s in range(num_slices):
        for i in range(num_RUs):
            for k in range(num_UEs):
                for b in range(num_RBs):
                    # Tính độ lợi kênh với Rayleigh fading và num_antennas
                    channel_gain = path_loss_linear[i, k] / noise_power_watts
                    h = np.sqrt(channel_gain) * np.sqrt(1 / 2) * (np.random.randn(num_antennas) + 1j * np.random.randn(num_antennas))
                
                    # Lưu độ lợi kênh vào ma trận
                    gain[i, b, s, k] = norm(h, 2)**2
    #print("gain: ", gain)
    return gain
