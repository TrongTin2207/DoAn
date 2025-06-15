import gen_RU_UE
import wireless
import RAN_topo
import solving
import other_function
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from validate import validate_long_term_solution, validate_short_term_solution, ValidationLogger

# =======================================================
# ============== Tham số mô phỏng =======================
# =======================================================
num_RUs = 3                            # Số lượng RU (bao gồm RU ở tâm)
num_DUs = 2                             # Số lượng DU
num_CUs = 2                             # Số lượng CU
num_UEs = 5                             # Tổng số lượng user cho tất dịch vụ (eMBB, mMTC, URLLC)
num_RBs = 5                             # Số lượng của RBs
num_antennas = 8                        # Số lượng anntenas
num_slices = 2                          # Số lượng loại dịch vụ - Changed to 2 to use all slice types

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
    slices = ["eMBB", "ULLRC"]              # Tập các loại slice

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

# Create a validation logger
validation_logger = ValidationLogger()

print("==== ORAN_MAPPING ====")
logger = other_function.Stopwatch("ORAN_MAPPING", silent=False)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename_prefix = f"{SAVE_PATH}/{timestamp}"
filename_problem = f"{filename_prefix}_problem.pkl.gz"
filename_solution = f"{filename_prefix}_solution"
filename_validation = f"{filename_prefix}_validation.txt"

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

    # Open validation log file
    validation_log_file = open(filename_validation, 'w')

    logger.add("[solve] Start solving...")
    for f in range(num_frame):
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Preparation")
        validation_log_file.write(f"\n\n======= FRAME {f+1} VALIDATION =======\n\n")
        
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
        if num_slices == 2:
            bandwidth_thresholds = {
                'eMBB': 6,      # For regular users (1-6 Mbps)
                'ULLRC': float('inf')  # For high-priority traffic (>6 Mbps)
            }
            # Create and assign slices with more sophisticated logic
            slice_names, R_min_array, D_j, D_m = gen_RU_UE.create_and_assign_slices_with_thresholds(
                num_UEs, D_j_random_list, D_m_random_list, R_min_random_list[:num_slices], bandwidth_thresholds
            )
            R_min = R_min_array  # Use the array directly
            
            # Convert slice names to mapping matrix
            slice_mapping = np.zeros((num_slices, num_UEs), dtype=int)
            for i, name in enumerate(slice_names):
                slice_mapping[0 if name == "eMBB" else 1, i] = 1
        else:
            # Use original simple mapping for single slice
            slice_mapping, D_j, D_m, R_min = gen_RU_UE.gen_mapping_and_requirements(
                num_UEs, num_slices, D_j_random_list, D_m_random_list, R_min_random_list
            )

        # Calculate channel gain
        gain = wireless.channel_gain(distances_RU_UE, num_slices, num_RUs, num_UEs, num_RBs, num_antennas, 
                                    path_loss_ref, path_loss_exp, noise_power_watts)
        other_function.save_object(
            f"{filename_solution}_gain_f{f}.pkl.gz",
            (gain)
        )

        # Long-term solution: solve global optimization
        validation_log_file.write("\n===== LONG-TERM SOLUTION VALIDATION =====\n")
        validation_log_file.write(f"Network Parameters:\n")
        validation_log_file.write(f"- Number of UEs: {num_UEs}\n")
        validation_log_file.write(f"- Number of RUs: {num_RUs}\n")
        validation_log_file.write(f"- Total available power per RU: {P_i} W\n")
        validation_log_file.write(f"- Resource blocks bandwidth: {rb_bandwidth} Hz\n")
        validation_log_file.write(f"- Minimum rate requirements: {R_min} bps\n")
        validation_log_file.write(f"- DU/CU resource limits: {A_j}/{A_m}\n")
        validation_log_file.write(f"- Resource demands: DU={D_j}, CU={D_m}\n")
        validation_log_file.write(f"- Slice mapping matrix:\n{slice_mapping}\n")
        validation_log_file.write(f"- Maximum channel gain: {np.max(gain)}\n")
        validation_log_file.write(f"- Minimum channel gain: {np.min(gain)}\n")
        
        long_term_result = solving.long_term(
            num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, 
            P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, 
            l_ru_du, l_du_cu, epsilon, gamma, slice_mapping,
            logger=logger  # Pass logger to get solver messages
        )

        # Check if any results are None before unpacking
        if any(x is None for x in long_term_result):
            logger.add(f"[solve] Frame {f+1}: No feasible solution found in long-term!")
            validation_log_file.write("\nLong-term solution failed with the following violations:\n")
            
            # Calculate theoretical maximum achievable rate
            max_possible_rate = np.sum([rb_bandwidth * np.log2(1 + np.max(gain) * P_i[0] / noise_power_watts) 
                                      for _ in range(num_RBs)])
            validation_log_file.write(f"- Maximum theoretically achievable rate: {max_possible_rate/1e6:.2f} Mbps\n")
            
            
            # Check resource constraints
            total_du_demand = np.sum([D_j[k] for k in range(num_UEs)])
            total_cu_demand = np.sum([D_m[k] for k in range(num_UEs)])
            validation_log_file.write(f"- Total DU resource demand: {total_du_demand} (limit: {A_j})\n")
            validation_log_file.write(f"- Total CU resource demand: {total_cu_demand} (limit: {A_m})\n")
            
            # Check topology constraints
            validation_log_file.write("- RU-DU-CU connectivity status:\n")
            for i in range(num_RUs):
                du_connections = sum(l_ru_du[i])
                if du_connections == 0:
                    validation_log_file.write(f"  RU{i} has no DU connections\n")
                for j in range(num_DUs):
                    if l_ru_du[i][j] == 1:
                        cu_connections = sum(l_du_cu[j])
                        if cu_connections == 0:
                            validation_log_file.write(f"  DU{j} (connected to RU{i}) has no CU connections\n")
            continue

        pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk = long_term_result

        # Check solution and save results
        if (pi_sk is None):
            logger.add(f"[solve] Frame {f+1}: No feasible solution found!")
            validation_log_file.write("No feasible long-term solution found.\n")
            continue
        
        # Validate long-term solution
        valid_long_term, rates_long_term = validate_long_term_solution(
            num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs,
            P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m,
            l_ru_du, l_du_cu, epsilon, gamma, slice_mapping,
            pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk,
            logger=validation_logger
        )
        validation_log_file.write(f"\nLong-term validation result: {'PASSED' if valid_long_term else 'FAILED'}\n")
        for log in validation_logger.get_logs():
            validation_log_file.write(f"{log}\n")
        validation_logger.logs = []  # Clear logs for next validation
            
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Long-term save")
        other_function.save_object(
            f"{filename_solution}_longterm_f{f}.pkl.gz",
            (pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk)
        )
        logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term")

        # Extract optimization results for short-term use
        arr_pi_sk, arr_z_ib_sk, arr_p_ib_sk, arr_mu_ib_sk, arr_phi_i_sk, arr_phi_j_sk, arr_phi_m_sk = other_function.extract_optimization_results(
            pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk
        )

        for t in range(time_slot):
            logger.add(f"[solve] Solve frame {f+1} of {num_frame}: Short-term time_slot={t+1} of {time_slot}: Preparation")
            validation_log_file.write(f"\n===== SHORT-TERM SOLUTION VALIDATION - FRAME {f+1}, TIME SLOT {t+1} =====\n")
            
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

            # Short-term optimization with None checks
            validation_log_file.write(f"\nShort-term optimization parameters:\n")
            validation_log_file.write(f"- New UE distances:\n{short_distances_RU_UE}\n")
            validation_log_file.write(f"- Channel gains summary: min={np.min(short_gain)}, max={np.max(short_gain)}\n")
            
            # Try multiple power allocation strategies if initial solution fails
            power_scale_factors = [1.0, 0.9, 0.8, 0.7]
            short_term_result = None
            
            for power_scale in power_scale_factors:
                scaled_P_i = [p * power_scale for p in P_i] if isinstance(P_i, list) else P_i * power_scale
                
                short_term_result = solving.short_term(
                    num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, scaled_P_i,
                    short_gain, R_min, epsilon, arr_pi_sk, arr_phi_i_sk,
                    logger=logger
                )
                
                if not any(x is None for x in short_term_result):
                    validation_log_file.write(f"\nFound feasible solution with {power_scale*100}% power allocation\n")
                    break
            
            if any(x is None for x in short_term_result):
                validation_log_file.write("\nShort-term solution failed with all power scaling attempts\n")
                # Add detailed failure analysis code...
                continue
                
            short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk, short_total_R_sk = short_term_result
            
            # Validate short-term solution
            if short_pi_sk is not None:
                valid_short_term, rates_short_term = validate_short_term_solution(
                    num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth,
                    P_i, short_gain, R_min, epsilon, arr_pi_sk, arr_phi_i_sk,
                    short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk,
                    logger=validation_logger
                )
                validation_log_file.write(f"\nShort-term validation result: {'PASSED' if valid_short_term else 'FAILED'}\n")
                for log in validation_logger.get_logs():
                    validation_log_file.write(f"{log}\n")
                validation_logger.logs = []  # Clear logs for next validation
            else:
                validation_log_file.write("No feasible short-term solution found.\n")
            
            # Save short-term results
            other_function.save_object(
                f"{filename_solution}_shortterm_f{f}_t{t}.pkl.gz",
                (short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk, short_total_R_sk)
            )

        # Adjust number of UEs for next frame (with random variation)
        num_UEs = max(num_UEs + np.random.randint(-delta_num_UE, delta_num_UE), 1)
        logger.add(f"[solve] Frame {f+1} completed. Next frame will have {num_UEs} UEs.")

    validation_log_file.close()
    logger.stop()
    print(f"Simulation completed successfully. Validation logs saved to {filename_validation}")
    

# Kiểm tra và chạy hàm main
if __name__ == "__main__":
    main()