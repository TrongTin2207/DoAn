import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

#Adding Coordinates constraint
#Threshold of Handover


def gen_coordinates_RU(num_RUs, radius):
    circle_RU_out = radius * 0.65
    angles = np.linspace(0, 2 * np.pi, num_RUs - 1, endpoint=False) 
    x = np.concatenate(([0], circle_RU_out * np.cos(angles)))  
    y = np.concatenate(([0], circle_RU_out * np.sin(angles)))  
    coordinates_RU = list(zip(x, y)) 
    return coordinates_RU

def gen_coordinates_UE(num_UEs, radius_in, radius_out):
    angles = np.random.uniform(0, 2 * np.pi, num_UEs)
    r = np.random.uniform(radius_in, radius_out, num_UEs)
    x = r * np.cos(angles)
    y = r * np.sin(angles)
    coordinates_UE = list(zip(x, y))  
    return coordinates_UE

def adjust_coordinates_UE(coordinates_UE, delta_coordinate, radius_in, radius_out):
    # Khởi tạo seed cho ngẫu nhiên để kết quả có thể tái tạo
    new_coordinates_UE = []
    
    for x, y in coordinates_UE:
        # Tạo độ lệch ngẫu nhiên trong khoảng [-delta_coordinate, delta_coordinate] cho cả x và y
        delta_x = np.random.uniform(-delta_coordinate, delta_coordinate)
        delta_y = np.random.uniform(-delta_coordinate, delta_coordinate)
        
        # Tọa độ mới sau khi thêm độ lệch
        new_x = x + delta_x
        new_y = y + delta_y
        
        #Tính khoảng cách từ tâm đảm bảo UE nằm trong các khoảng
        distance_from_center = np.sqrt(new_x**2 + new_y**2)

        #Thêm các ràng buộc về khoảng
        if distance_from_center < radius_in:
            #Nếu như UE trong bán kính vùng trong, chỉnh sao cho nằm trong khoảng trong
            scale_factor = radius_in / distance_from_center
            new_x *= scale_factor
            new_y *= scale_factor
        elif distance_from_center < radius_out:
            #Nếu như UE trong bán kính vùng ngoài, chỉnh sao cho nằm trong khoảng ngoài
            scale_factor = radius_out / distance_from_center
            new_x *= scale_factor
            new_y *= scale_factor
        # Thêm tọa độ mới vào danh sách
        new_coordinates_UE.append((new_x, new_y))
    
    return new_coordinates_UE

#Hàm tính Handover
def determine_handover(distances_RU_UE, current_RU_assignents, handover_margin=50):
    num_RUs, num_UEs = distances_RU_UE.shape
    new_RU_assignments = np.copy(current_RU_assignents)
    handover_events = []

    for ue_id in range (num_UEs):
        #Lấy RU được chọn hiện tại
        current_ru = np.where(current_RU_assignents[:, ue_id]==1)[0]
        if len(current_ru) == 0:
            #Nếu như không có RU nào được chọn thì chọn 1 RU
            best_ru = np.argmin(distances_RU_UE[:, ue_id])
            new_RU_assignments[best_ru, ue_id] = 1
            continue

        current_ru = current_ru[0]
        current_distance = distances_RU_UE[current_ru, ue_id]

        #Tìm RU gần nhất
        closest_ru = np.argmin(distances_RU_UE[:, ue_id])
        closest_distance = distances_RU_UE[closest_ru, ue_id]

        #Kiểm tra nếu handover có thể xảy ra dựa vào khoảng cách được cải thiện
        if (current_ru != closest_ru) and (current_distance - closest_distance > handover_margin):
            new_RU_assignments[current_ru, ue_id] = 0
            new_RU_assignments[closest_ru, ue_id] = 1
            handover_events.append((ue_id, current_ru,closest_ru))
        
        return new_RU_assignments,handover_events
#Hàm tính khoảng cách (d)
def calculate_distances(coordinates_RU, coordinates_UE, num_RUs, num_UEs):
    distances_RU_UE = np.zeros((num_RUs, num_UEs))
    for i in range(num_RUs):
        for j in range(num_UEs):
            x_RU, y_RU = coordinates_RU[i]
            x_UE, y_UE = coordinates_UE[j]
            distances_RU_UE[i, j] = np.sqrt((x_RU - x_UE)**2 + (y_RU - y_UE)**2)
    return distances_RU_UE

def create_and_assign_slices(num_UEs, num_slices, D_j, D_m, R_min):  
    # Initialize lists to store each attribute
    names = []
    R_min_values = []
    D_j_values = []
    D_m_values = []
    
    # First, generate bandwidth requirements for all users
    # We'll use R_min (minimum data rate) as the bandwidth metric
    # Generate random R_min values for all UEs
    all_R_min = np.random.choice(R_min, size=num_UEs)
    
    # Sort UEs by their bandwidth requirements
    sorted_indices = np.argsort(all_R_min)
    
    # Determine thresholds for different slices
    if num_slices == 3:  # If we have three slices (eMBB, ULLRC, mMTC)
        # Calculate how many UEs should be in each slice
        ues_per_slice = num_UEs // 3
        
        # Assign slices based on bandwidth requirements:
        # - mMTC: Lower third of bandwidth requirements (IoT devices, sensors)
        # - eMBB: Middle third of bandwidth requirements (regular users)
        # - ULLRC: Upper third of bandwidth requirements (latency-critical applications)
        
        for i in range(num_UEs):
            position = np.where(sorted_indices == i)[0][0]  # Find position in sorted array
            
            if position < ues_per_slice:  # Lower third - mMTC
                names.append("mMTC")
                R_min_values.append(all_R_min[i] * 0.5)  # Lower data rate for mMTC
            elif position < 2 * ues_per_slice:  # Middle third - eMBB
                names.append("eMBB")
                R_min_values.append(all_R_min[i])  # Standard data rate for eMBB
            else:  # Upper third - ULLRC
                names.append("ULLRC")
                R_min_values.append(all_R_min[i] * 2)  # Higher data rate for ULLRC
            
            # Assign D_j and D_m values (can be customized further based on slice needs)
            D_j_values.append(np.random.choice(D_j))
            D_m_values.append(np.random.choice(D_m))
    
    elif num_slices == 2:  # If we have two slices
        # Calculate how many UEs should be in each slice
        ues_per_slice = num_UEs // 2
        
        for i in range(num_UEs):
            position = np.where(sorted_indices == i)[0][0]  # Find position in sorted array
            
            if position < ues_per_slice:  # Lower half - e.g., mMTC
                names.append("mMTC")
                R_min_values.append(all_R_min[i] * 0.5)
            else:  # Upper half - e.g., eMBB
                names.append("eMBB")
                R_min_values.append(all_R_min[i])
            
            D_j_values.append(np.random.choice(D_j))
            D_m_values.append(np.random.choice(D_m))
    
    else:  # For any other number of slices
        # Divide UEs evenly among slices
        slice_size = num_UEs // num_slices
        remaining = num_UEs % num_slices
        
        for i in range(num_UEs):
            position = np.where(sorted_indices == i)[0][0]  # Find position in sorted array
            slice_idx = min(position // slice_size, num_slices - 1)
            
            # Assign a generic slice name
            names.append(f"Slice_{slice_idx}")
            
            # Adjust R_min based on slice index (higher index = higher bandwidth)
            adjustment_factor = 0.5 + (slice_idx / (num_slices - 1)) * 1.5
            R_min_values.append(all_R_min[i] * adjustment_factor)
            
            D_j_values.append(np.random.choice(D_j))
            D_m_values.append(np.random.choice(D_m))
    
    # Return separate arrays
    return names, R_min_values, D_j_values, D_m_values


# Alternative implementation with customizable bandwidth thresholds
def create_and_assign_slices_with_thresholds(num_UEs, D_j, D_m, R_min, bandwidth_thresholds=None):
    """
    Assign slices based on bandwidth requirements using custom thresholds.
    
    Parameters:
    - num_UEs: Number of user equipment
    - D_j, D_m, R_min: Lists of possible values for requirements
    - bandwidth_thresholds: Dictionary with slice names as keys and upper thresholds as values
                           e.g., {'mMTC': 2, 'eMBB': 5, 'ULLRC': float('inf')}
                           This means mMTC for <= 2 Mbps, eMBB for 2-5 Mbps, ULLRC for > 5 Mbps
    """
    # Initialize lists to store each attribute
    names = []
    R_min_values = []
    D_j_values = []
    D_m_values = []
    
    # Default thresholds if none provided
    if bandwidth_thresholds is None:
        bandwidth_thresholds = {
            'mMTC': 2,      # For low bandwidth requirements (<=2 Mbps)
            'eMBB': 10,     # For medium bandwidth requirements (2-10 Mbps)
            'ULLRC': float('inf')  # For high bandwidth requirements (>10 Mbps)
        }
    
    # Sort the thresholds by value
    sorted_slices = sorted(bandwidth_thresholds.items(), key=lambda x: x[1])
    
    # Generate random bandwidth requirements for each UE
    # This could be from actual bandwidth measurements or requirements
    bandwidths = np.random.uniform(0.5, 15, size=num_UEs)  # Random values between 0.5 and 15 Mbps
    
    # Assign slices based on bandwidth requirements
    for i in range(num_UEs):
        bandwidth = bandwidths[i]
        
        # Find appropriate slice based on thresholds
        assigned_slice = sorted_slices[-1][0]  # Default to highest slice
        for slice_name, threshold in sorted_slices:
            if bandwidth <= threshold:
                assigned_slice = slice_name
                break
        
        names.append(assigned_slice)
        
        # Adjust R_min based on slice type
        if assigned_slice == 'mMTC':
            R_min_values.append(np.random.choice(R_min) * 0.5)
        elif assigned_slice == 'ULLRC':
            R_min_values.append(np.random.choice(R_min) * 2)
        else:  # eMBB or other
            R_min_values.append(np.random.choice(R_min))
        
        D_j_values.append(np.random.choice(D_j))
        D_m_values.append(np.random.choice(D_m))
    
    # Return separate arrays
    return names, R_min_values, D_j_values, D_m_values



def plot_save_network(coordinates_RU, coordinates_UE, radius_in, radius_out):
    circle_in = plt.Circle((0, 0), radius_in, color='gray', fill=False, linestyle='--', label='Inner Radius')
    circle_out = plt.Circle((0, 0), radius_out, color='black', fill=False, linestyle='--', label='Outer Radius')
    
    plt.gca().add_artist(circle_in)
    plt.gca().add_artist(circle_out)
    
    for (x, y) in coordinates_RU:
        plt.scatter(x, y, color='green', marker='^', s=100, label='RU' if (x, y) == (0, 0) else "")
    
    for index, (x, y) in enumerate(coordinates_UE):
        plt.scatter(x, y, color='blue', marker='o')
        if index == 0:  # Chỉ chú thích cho UE đầu tiên
            plt.scatter(x, y, color='blue', marker='o', label='UE')

    plt.xlim(-radius_out * 1.2, radius_out * 1.2)
    plt.ylim(-radius_out * 1.2, radius_out * 1.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.title("Network with RU and UE")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right')
    
    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)
    fig_name = os.path.join(result_dir, f"network_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    plt.savefig(fig_name)
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def gen_mapping_and_requirements(num_UEs, num_slices, D_j_random_list, D_m_random_list, R_min_random_list):
    # Khởi tạo ma trận ánh xạ với tất cả giá trị là 0
    slice_mapping = np.zeros((num_slices, num_UEs), dtype=int)
    
    # Danh sách để lưu yêu cầu tài nguyên cho mỗi UE
    D_j_list = []
    D_m_list = []
    R_min_list = []
    
    if num_slices == 1:
        # Nếu chỉ có một slice, tất cả UE được ánh xạ vào slice 0
        slice_mapping[0, :] = 1
    else:
        for ue in range(num_UEs):
            # Mỗi UE chỉ được yêu cầu một loại slice, chọn ngẫu nhiên một slice
            chosen_slice = np.random.randint(0, num_slices)
            slice_mapping[chosen_slice][ue] = 1
            
    # Tạo ngẫu nhiên các giá trị D_j, D_m, R_min cho mỗi UE
    D_j_list = np.random.choice(D_j_random_list, size=num_UEs).tolist()
    D_m_list = np.random.choice(D_m_random_list, size=num_UEs).tolist()
    R_min_list = np.random.choice(R_min_random_list, size=num_UEs).tolist()
    
    return slice_mapping, D_j_list, D_m_list, R_min_list

def calculate_rsrp(gain, p_ib, num_RUs, num_UEs):
    rsrp_matrix = np.zeros((num_RUs, num_UEs))
    for i in range(num_RUs):
        for k in range(num_UEs):
            allocated_power = np.sum(p_ib[i, :, k])
            rsrp_linear = gain[i, k] * allocated_power

            # Use a numeric approach instead of boolean evaluation
            try:
                # Try to handle the case when values are numeric
                if isinstance(rsrp_linear, (int, float)) and rsrp_linear > 0:
                    rsrp_matrix[i, k] = 10 * np.log10(rsrp_linear) + 30  # dBm
                else:
                    rsrp_matrix[i, k] = -float('inf')
            except:
                # Alternative approach for symbolic variables
                # Use mathematical operations instead of conditionals
                epsilon = 1e-10
                # A numeric approximation approach:
                safe_value = max(rsrp_linear, epsilon)  # This might need adjustment based on your library
                rsrp_matrix[i, k] = 10 * np.log10(safe_value) + 30
    
    return rsrp_matrix

def determine_handover_rsrp(rsrp_matrix, current_RU_assignments, rsrp_threshold = -110, hysteresis = 3):
    num_RUs, num_UEs = rsrp_matrix.shape
    new_RUs_assignments = np.copy(current_RU_assignments)
    handover_events = []

    for ue_id in range(num_UEs):
        #Chọn RU hiện tại
        current_ru_indices = np.where(current_RU_assignments[:, ue_id] == 1)[0]

        if len(current_ru_indices) == 0:
            #Nhiệm vụ ban đầu nếu hiện tại không có RU nào được chỉ định
            best_ru = np.argmax(rsrp_matrix[:, ue_id])
            if rsrp_matrix[best_ru, ue_id] >= rsrp_threshold:
                new_RUs_assignments[best_ru, ue_id] = 1
            continue

        current_ru = current_ru_indices[0]
        current_rsrp = rsrp_matrix[current_ru, ue_id]

        #Kiểm tra nếu RSRP hiện tại dưới threshold (nhu cầu handover được kích hoạt)
        if current_rsrp < rsrp_threshold:
            #Tìm RU tốt nhất dựa trên RSRP
            candidate_rus = np.argsort(rsrp_matrix[:, ue_id])[::-1] #Xếp từ trên xuống dưới
            for candidate_ru in candidate_rus:
                candidate_rsrp = rsrp_matrix[candidate_ru, ue_id]

                #Kiểm tra nếu RU candidate trên mức threshold và tốt hơn so với hiện tại
                if candidate_rsrp >= rsrp_threshold and candidate_ru != current_ru:
                    new_RUs_assignments[current_ru, ue_id] = 0
                    new_RUs_assignments[candidate_ru, ue_id] = 1
                    handover_events.append((ue_id, current_ru, candidate_ru, current_rsrp, candidate_rsrp))
                    break
            else:
                #Kiểm tra xem có RU nào khác có RSRP tốt hơn (với trễ)
                best_ru = np.argmax(rsrp_matrix[:, ue_id])
                best_rsrp = rsrp_matrix[best_ru, ue_id]

                #Áp trễ để chặn ping-pong handover
                if best_ru != current_ru and best_rsrp > current_rsrp + hysteresis:
                    new_RUs_assignments[current_ru, ue_id] = 0
                    new_RUs_assignments[best_ru, ue_id] = 1
                    handover_events.append((ue_id, current_ru, best_ru, current_rsrp, best_rsrp))
        return new_RUs_assignments, handover_events

def visualize_handovers(coordiantes_RU, coordinates_UE_before, coordinates_UE_after, handover_events, radius_in, radius_out, frame, time_slot):
    plt.figure(figsize=(12,10))

    #Vẽ ranh giới mạng
    circle_in = plt.Circle((0,0), radius_in, color='gray', fill = False, linestyle = '--', label = 'Inner Radius')
    circle_out = plt.Circle((0,0), radius_out, color='black', fill = False, linestyle = '--', label = 'Outer Radius')
    plt.gca().add_artist(circle_in)
    plt.gca().add_artist(circle_out)

    #Plot RUs
    for i, (x,y) in enumerate(coordiantes_RU):
        plt.scatter(x,y,color = 'green', marker='^',s=200, label='RU' if i ==0 else "")
        plt.text(x, y+20, f"RU{i+1}", fontsize = 10, ha = 'center')
    
    #Plot UEs trước khi di chuyển
    for i, (x,y) in enumerate(coordinates_UE_before):
        if i < len(coordinates_UE_after): #Đảm bảo UE tồn tại ở 2 trạng thái
            plt.scatter(x, y, color='blue', marker='o', alpha = 0.3)

        #Plot UEs sau khi di chuyển
        x_new, y_new = coordinates_UE_after[i]
        plt.scatter(x_new, y_new, color='blue', marker='o', label='UE' if i == 0 else "")
        plt.text(x_new, y_new+10, f"UE{i+1}", fontsize = 9, ha='center')
        
        #Vẽ vector di chuyển
        plt.arrow(x, y, x_new-x, y_new-y, color='gray', width=1, head_width=10, head_length=10, alpha=0.7, lengt_includes_head=True)
    
    #Highlight sự kiện handover
    for event in handover_events:
        ue_id, old_ru, new_ru = event[0], event[1], event[2]
        if ue_id < len(coordinates_UE_after):
            x_ue, y_ue = coordinates_UE_after[ue_id]
            x_old_ru, y_old_ru = coordiantes_RU[old_ru]
            x_new_ru, y_new_ru = coordiantes_RU[new_ru]

            #Vẽ đường từ RU cũ đến UE (ngắt kết nối)
            plt.plot([x_old_ru, x_ue], [y_old_ru, y_ue], 'r--', linewidth=1.5, alpha=0.7)

            #Vẽ đường từ UE đến RU mới (kết nối)
            plt.plot([x_new_ru, x_ue], [y_new_ru, y_ue], 'g-', linewidth=1.5, alpha=0.7)

            #Thêm annotation
            plt.annotate(f"HO: UE{ue_id+1}\nRU{old_ru+1}->RU{new_ru+1}",
                         xy=(x_ue, y_ue), xytext=(x_ue+30, y_ue+30),
                         arrowprops=dict(arrowstyle="->", color='red'))
        
    plt.xlim(-radius_out * 1.2, radius_out * 1.2)
    plt.ylim(-radius_out * 1.2, radius_out * 1.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(alpha=0.3)
    plt.title(f"Network with UE Movement and Handovers (Frame {frame+1}, Time Slot {time_slot+1})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right')
    
    # Save figure
    result_dir = "./result"
    os.makedirs(result_dir, exist_ok=True)
    fig_name = f"{result_dir}/handover_visualization_f{frame}_t{time_slot}.pdf"
    plt.savefig(fig_name)
    plt.close()