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

def adjust_coordinates_UE(coordinates_UE, delta_coordinate):
    # Khởi tạo seed cho ngẫu nhiên để kết quả có thể tái tạo
    new_coordinates_UE = []
    
    for x, y in coordinates_UE:
        # Tạo độ lệch ngẫu nhiên trong khoảng [-delta_coordinate, delta_coordinate] cho cả x và y
        delta_x = np.random.uniform(-delta_coordinate, delta_coordinate)
        delta_y = np.random.uniform(-delta_coordinate, delta_coordinate)
        
        # Tọa độ mới sau khi thêm độ lệch
        new_x = x + delta_x
        new_y = y + delta_y
        
        # Thêm tọa độ mới vào danh sách
        new_coordinates_UE.append((new_x, new_y))
    
    return new_coordinates_UE

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
