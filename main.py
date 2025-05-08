import gen_RU_UE
import wireless
import RAN_topo
import solving
import benchmark
import other_function
import numpy as np
import datetime
import os

# =======================================================
# ============== Tham số mô phỏng =======================
# =======================================================
num_RUs = 4                             # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 3                             # Số lượng DU
num_CUs = 3                             # Số lượng CU
num_UEs = 5                             # Tổng số lượng user cho tất dịch vụ (eMBB, mMTC, URLLC)
num_RBs = 5                             # Số lượng của RBs
num_antennas = 8                        # Số lượng anntenas
num_slices = 2                          # Số lượng loại dịch vụ (Changed to 2 to include uRLLC)

radius_in = 100                         # Bán kính vòng tròn trong (km)
radius_out = 1000                       # Bán kính vòng tròn ngoài (km)

rb_bandwidth = 180e3                                # Băng thông của mỗi RBs (Hz)
# Maximum transmission power
max_tx_power_dbm = 43                               # dBm
max_tx_power_mwatts = 10**((max_tx_power_dbm)/10)   # Công suất tại mỗi RU (mW)
noise_power_watts = 1e-10                           # Công suất nhiễu (mW)   

epsilon = 1e-10                                     # Giá trị nhỏ ~ 0

P_j_random_list = [max_tx_power_mwatts]

path_loss_ref = 128.1
path_loss_exp = 37.6

if num_slices == 1:
    slices = ["eMBB"]                               # Tập các loại slice
else:
    slices = ["uRLLC", "eMBB"]                      # Tập các loại slice (Changed to uRLLC and eMBB)

# Speed of light parameters
speed_of_light = 3e8                    # Speed of light in meters/second (m/s)
speed_of_light_km_ms = speed_of_light / 1e6  # Speed of light in km/ms for propagation delay calculation

# Latency-related parameters
max_latency = 1.0                       # Maximum allowed latency for uRLLC in ms
L_cu = 0.02                             # Processing time at CU (ms)
L_du = 0.05                             # Processing time at DU (ms)
rho_du = np.array([0.8, 0.6])           # Queuing parameter for slices [uRLLC, eMBB]
mu_s = np.array([100, 50])              # Service rate for slices [uRLLC, eMBB] (packets/ms)
lambda_s = np.array([80, 30])           # Arrival rate for slices [uRLLC, eMBB] (packets/ms)

# Packet parameters
packet_length = 1500 * 8                # Packet length in bits (1500 bytes * 8 bits/byte)
packet_processing_time_du = 0.05        # Processing time for a packet at DU (ms)
packet_processing_time_cu = 0.02        # Processing time for a packet at CU (ms)

D_j_random_list = [5]                                   # Các loại yêu cầu tài nguyên của node DU j 
D_m_random_list = [5]                                   # Các loại yêu cầu tài nguyên của node CU m 

A_j_random_list = [50]                                  # Các loại tài nguyên của node DU j
A_m_random_list = [50]                                  # Các loại tài nguyên của node CU m

R_min_random_list = [1e6]                               # Các loại yêu cầu Data rate ngưỡng

delta_coordinate = 5                               # Sai số toạ độ của UE
delta_num_UE = 5                                    # Sai số số lượng UE

time_slot = 5                                       # Số lượng time slot trong 1 frame
num_frame = 5

gamma = 0.8                                         # Hệ số tối ưu

SAVE_PATH = "./result" # SAVE PATH
os.makedirs(SAVE_PATH, exist_ok=True)

print("==== ORAN_MAPPING ====")
logger = other_function.Stopwatch("ORAN_MAPPING", silent=False)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename_prefix = f"{SAVE_PATH}/{timestamp}"
filename_problem = f"{filename_prefix}_problem.pkl.gz"
filename_solution = f"{filename_prefix}_solution"

# ===========================================
# ============== Main =======================
# ===========================================

def main():
    logger.start()
    logger.add("[instance] Generate instance")
    global num_UEs
    seed = 42
    np.random.seed(seed)

    # Toạ toạ độ RU
    coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)

    # Tạo mạng RAN
    G = RAN_topo.create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list)

    # Danh sách tập các liên kết trong mạng
    l_ru_du, l_du_cu = RAN_topo.get_links(G)

    # Tập các capacity của các node DU, CU
    P_i, A_j, A_m = RAN_topo.get_node_cap(G)
    
    # Save instances
    logger.add("[instance] Save instance")
    other_function.save_object(
        f"{filename_prefix}_physicalnet.pkl.gz",
        G
    )
    other_function.save_object(
        f"{filename_prefix}_coords_RUs.pkl.gz",
        coordinates_RU
    )

    logger.add("[solve] Start solving...")
    for f in range(num_frame):
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Preparation")
        
        # Tạo yêu cầu từng UE
        slice_mapping, D_j, D_m, R_min = gen_RU_UE.gen_mapping_and_requirements(num_UEs, num_slices, D_j_random_list, D_m_random_list, R_min_random_list)

        # Tạo toạ độ UE
        coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)
        other_function.save_object(
            f"{filename_prefix}_coords_UEs_f{f}.pkl.gz",
            coordinates_UE
        )

        # Ma trận khoảng cách của UE - RU
        distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)
        
        # Create d_sk matrix (distance for each UE in each slice)
        d_sk = np.zeros((num_slices, num_UEs))
        for s in range(num_slices):
            for k in range(num_UEs):
                if slice_mapping[s, k] == 1:
                    # Find minimum distance from UE k to any RU
                    d_sk[s, k] = np.min(distances_RU_UE[:, k])

        # Tính gain cho mạng
        gain = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, path_loss_ref, path_loss_exp, noise_power_watts)
        other_function.save_object(
            f"{filename_solution}_gain_f{f}.pkl.gz",
            (gain)
        )

        # Gọi làm long-term: giải bài toán toàn cục
        pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk = solving.long_term(
            num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, 
            P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, 
            l_ru_du, l_du_cu, epsilon, gamma, slice_mapping,
            c=speed_of_light, d_sk=d_sk, max_latency=max_latency,
            L_cu=L_cu, L_du=L_du, rho_du=rho_du, mu_s=mu_s, lambda_s=lambda_s,
            logger=logger
        )

        # In ra nghiệm
        if (pi_sk is None):
            logger.add(f"[solve] Frame {f+1}: Long-term optimization failed, skipping to next frame")
            continue
            
        # benchmark.print_results(pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk)
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Long-term save")
        other_function.save_object(
            f"{filename_solution}_longterm_f{f}.pkl.gz",
            (pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk)
        )
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term")

        for t in range(time_slot):
            logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term time_slot={t+1} of {time_slot}: Preparation")
            # UE di chuyển (có toạ độ mới)
            short_coordinates_UE = gen_RU_UE.adjust_coordinates_UE(coordinates_UE, delta_coordinate)
            other_function.save_object(
                f"{filename_prefix}_coords_UEs_f{f}_t{t}.pkl.gz",
                short_coordinates_UE
            )
            # Khoảng cách mới từ UE - RU sau khi di chuyển
            short_distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, short_coordinates_UE, num_RUs, num_UEs)
            
            # Update d_sk for the new distances
            short_d_sk = np.zeros((num_slices, num_UEs))
            for s in range(num_slices):
                for k in range(num_UEs):
                    if slice_mapping[s, k] == 1:
                        # Find minimum distance from UE k to any RU
                        short_d_sk[s, k] = np.min(short_distances_RU_UE[:, k])

            # Tính gain cho kênh truyền thay đổi
            short_gain = wireless.channel_gain(short_distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, 
                path_loss_ref, path_loss_exp, noise_power_watts)
            other_function.save_object(
                f"{filename_solution}_gain_f{f}_t{t}.pkl.gz",
                (short_gain)
            )

            # Chuyển kết quả thành mảng
            arr_pi_sk, arr_z_ib_sk, arr_p_ib_sk, arr_mu_ib_sk, arr_phi_i_sk, arr_phi_j_sk, arr_phi_m_sk = other_function.extract_optimization_results(pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk)

            # Tối ưu Short-term
            short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk, short_total_R_sk = solving.short_term(
                num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, 
                P_i, short_gain, R_min, epsilon, arr_pi_sk, arr_phi_i_sk,
                c=speed_of_light, d_sk=short_d_sk, max_latency=max_latency,
                L_cu=L_cu, L_du=L_du, rho_du=rho_du, mu_s=mu_s, lambda_s=lambda_s,
                logger=logger
            )
            
            if short_pi_sk is None:
                logger.add(f"[solve] Frame {f+1}, Time Slot {t+1}: Short-term optimization failed")
                continue
                
            # Save short-term results
            other_function.save_object(
                f"{filename_solution}_shortterm_f{f}_t{t}.pkl.gz",
                (short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk, short_total_R_sk)
            )

            # In kết quả short-term
            # benchmark.print_result_short_term(short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk)

        # Update number of UEs for next frame
        num_UEs = max(num_UEs + np.random.randint(-delta_num_UE, delta_num_UE), 1)  # Ensure at least 1 UE
    
    logger.add(f"[solve] All frames completed successfully")
    logger.stop()

# Kiểm tra và chạy hàm main
if __name__ == "__main__":
    main()