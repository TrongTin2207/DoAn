import gen_RU_UE
import wireless
import RAN_topo
import solving
import benchmark
import other_function
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

# =======================================================
# ============== Tham số mô phỏng =======================
# =======================================================
num_RUs = 4                             # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 3                             # Số lượng DU
num_CUs = 3                             # Số lượng CU
num_UEs = 5                             # Tổng số lượng user cho tất dịch vụ (eMBB, mMTC, URLLC)
num_RBs = 5                             # Số lượng của RBs
num_antennas = 8                        # Số lượng anntenas
num_slices = 3                          # Số lượng loại dịch vụ - Changed to 3 to use all slice types

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

# Define slices based on num_slices
if num_slices == 1:
    slices = ["eMBB"]                               # Tập các loại slice
else:
    slices = ["eMBB", "ULLRC", "mMTC"]              # Tập các loại slice

speed_of_light = 3e8                    # Speed of light in meters/second (m/s)
speed_of_light_km_ms = speed_of_light / 1e6  # Speed of light in km/ms for propagation delay calculation

# Packet parameters (added)
packet_length = 1500 * 8                # Packet length in bits (1500 bytes * 8 bits/byte)
packet_processing_time_du = 0.05        # Processing time for a packet at DU (ms)
packet_processing_time_cu = 0.02        # Processing time for a packet at CU (ms)

D_j_random_list = [5]                                   # Các loại yêu cầu tài nguyên của node DU j 
D_m_random_list = [5]                                   # Các loại yêu cầu tài nguyên của node CU m 

A_j_random_list = [50]                                  # Các loại tài nguyên của node DU j
A_m_random_list = [50]                                  # Các loại tài nguyên của node CU m

R_min_random_list = [1e6, 2e6, 5e5]                     # Modified for different R_min values per slice type

delta_coordinate = 5                               # Sai số toạ độ của UE
delta_num_UE = 5                                   # Sai số số lượng UE

time_slot = 5                                      # Số lượng time slot trong 1 frame
num_frame = 5

gamma = 0.8                                        # Hệ số tối ưu

# Handover threshold parameter
handover_threshold = 50                            # Threshold distance for handover consideration

# Define coordinate constraints for RAN topology
coordinate_constraints = {
    'x_min': -radius_out,
    'x_max': radius_out,
    'y_min': -radius_out,
    'y_max': radius_out
}

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

    # Generate RU coordinates
    coordinates_RU = gen_RU_UE.gen_coordinates_RU(num_RUs, radius_out)

    # Create RAN topology with coordinate constraints (standard version, not handover version)
    G = RAN_topo.create_topo(
        num_RUs, num_DUs, num_CUs, 
        P_j_random_list, A_j_random_list, A_m_random_list,
        coordinate_constraints=coordinate_constraints, 
        RU_coordinates=coordinates_RU
    )

    # Draw the network topology with coordinates
    RAN_topo.draw_topo_coordinates(G)

    # Get links in network
    l_ru_du, l_du_cu = RAN_topo.get_links(G)

    # Get node capacities
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
        
        # Generate UE coordinates
        coordinates_UE = gen_RU_UE.gen_coordinates_UE(num_UEs, radius_in, radius_out)
        other_function.save_object(
            f"{filename_prefix}_coords_UEs_f{f}.pkl.gz",
            coordinates_UE
        )

        # Calculate distance matrix between UEs and RUs
        distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs)

        # Determine RU associations and handover candidates
        associations, handover_candidates = gen_RU_UE.determine_ru_associations(distances_RU_UE, handover_threshold)
        
        # Plot network with associations and handover candidates
        gen_RU_UE.plot_save_network(
            coordinates_RU, coordinates_UE, radius_in, radius_out, 
            associations, handover_candidates, handover_threshold
        )

        # Use enhanced slice assignment function instead of simple mapping
        if num_slices == 3:
            # Use the new slice assignment function with customized thresholds
            bandwidth_thresholds = {
                'mMTC': 1,      # For IoT devices with low bandwidth (<=1 Mbps)
                'eMBB': 6,      # For regular users (1-6 Mbps)
                'ULLRC': float('inf')  # For high-priority, low-latency traffic (>6 Mbps)
            }
            slice_names, R_min, D_j, D_m = gen_RU_UE.create_and_assign_slices_with_thresholds(
                num_UEs, D_j_random_list, D_m_random_list, R_min_random_list, bandwidth_thresholds)
            
            # Convert slice names to mapping matrix
            slice_mapping = np.zeros((num_slices, num_UEs), dtype=int)
            for i, name in enumerate(slice_names):
                if name == "eMBB":
                    slice_mapping[0, i] = 1
                elif name == "ULLRC":
                    slice_mapping[1, i] = 1
                elif name == "mMTC":
                    slice_mapping[2, i] = 1
        else:
            # Use default simple mapping for single slice
            slice_mapping, D_j, D_m, R_min = gen_RU_UE.gen_mapping_and_requirements(
                num_UEs, num_slices, D_j_random_list, D_m_random_list, R_min_random_list)

        # Calculate channel gain
        gain = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, 
                                    path_loss_ref, path_loss_exp, noise_power_watts)
        other_function.save_object(
            f"{filename_solution}_gain_f{f}.pkl.gz",
            (gain)
        )

        # Long-term solution: solve global optimization
        pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk = solving.long_term(
            num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, 
            P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, 
            l_ru_du, l_du_cu, epsilon, gamma, slice_mapping
        )

        # Check solution and save results
        if (pi_sk is None):
            logger.add(f"[solve] Frame {f+1}: No feasible solution found!")
            continue
            
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Long-term save")
        other_function.save_object(
            f"{filename_solution}_longterm_f{f}.pkl.gz",
            (pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk)
        )
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term")

        for t in range(time_slot):
            logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term time_slot={t+1} of {time_slot}: Preparation")
            
            # UE movement (new coordinates)
            short_coordinates_UE = gen_RU_UE.adjust_coordinates_UE(coordinates_UE, delta_coordinate)
            other_function.save_object(
                f"{filename_prefix}_coords_UEs_f{f}_t{t}.pkl.gz",
                short_coordinates_UE
            )
            
            # Calculate new RU-UE distances after movement
            short_distances_RU_UE = gen_RU_UE.calculate_distances(coordinates_RU, short_coordinates_UE, num_RUs, num_UEs)
            
            # Determine new associations and handover candidates
            short_associations, short_handover_candidates = gen_RU_UE.determine_ru_associations(
                short_distances_RU_UE, handover_threshold
            )
            
            # Plot network with updated positions and associations
            gen_RU_UE.plot_save_network(
                coordinates_RU, short_coordinates_UE, radius_in, radius_out, 
                short_associations, short_handover_candidates, handover_threshold
            )

            # Calculate new channel gain
            short_gain = wireless.channel_gain(
                short_distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, 
                path_loss_ref, path_loss_exp, noise_power_watts
            )
            other_function.save_object(
                f"{filename_solution}_gain_f{f}_t{t}.pkl.gz",
                (short_gain)
            )

            # Convert optimization results to arrays
            arr_pi_sk, arr_z_ib_sk, arr_p_ib_sk, arr_mu_ib_sk, arr_phi_i_sk, arr_phi_j_sk, arr_phi_m_sk = other_function.extract_optimization_results(
                pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk
            )

            # Short-term optimization
            short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk, short_total_R_sk = solving.short_term(
                num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, P_i, 
                short_gain, R_min, epsilon, arr_pi_sk, arr_phi_i_sk
            )
            
            # Save short-term results
            other_function.save_object(
                f"{filename_solution}_shortterm_f{f}_t{t}.pkl.gz",
                (short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk, short_total_R_sk)
            )

        # Adjust number of UEs for next frame (with random variation)
        num_UEs = max(num_UEs + np.random.randint(-delta_num_UE, delta_num_UE), 1)
        logger.add(f"[solve] Frame {f+1} completed. Next frame will have {num_UEs} UEs.")

    logger.stop()
    print("Simulation completed successfully.")
    

# Kiểm tra và chạy hàm main
if __name__ == "__main__":
    main()