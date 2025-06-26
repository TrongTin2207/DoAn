import datetime
import numpy as np
import matplotlib.pyplot as plt
import os

#Constraint Coordinate (RU (two RUs are 70% radius far) and UE distance only), CU and DU as Physical links (No need coordinates, as they have considered to be linked)

def gen_coordinates_RU(num_RUs, radius_RU, min_distance=400): #khong bi trung lan
    circle_RU_out = radius_RU * 0.65  # Bán kính tính toán vị trí các RU
    
    # RU trung tâm
    coordinates_RU = [(0, 0)]
    
    if num_RUs > 1:
        # Tạo các góc để phân bố RUs quanh vòng tròn
        angles = np.linspace(0, 2 * np.pi, num_RUs, endpoint=False)
        
        # Tính toán vị trí RU đảm bảo khoảng cách tối thiểu
        for angle in angles:
            x = circle_RU_out * np.cos(angle)
            y = circle_RU_out * np.sin(angle)
            
            # Kiểm tra khoảng cách để đảm bảo không chồng lấn quá mức
            valid_position = True
            for existing_x, existing_y in coordinates_RU:
                distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                if distance < min_distance:  # Kiểm tra khoảng cách tối thiểu
                    valid_position = False
                    break
            
            if valid_position:
                coordinates_RU.append((x, y))
    
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

def determine_ru_associations(distances_RU_UE, handover_threshold):
    num_RUs, num_UEs = distances_RU_UE.shape
    associations = np.zeros((num_RUs, num_UEs), dtype = int)
    handover_candidates=[]

    for ue_idx in range(num_UEs):
        #Tìm RU gần với UE này nhất
        closest_ru_idx = np.argmin(distances_RU_UE[:, ue_idx])
        min_distance = distances_RU_UE[closest_ru_idx, ue_idx]

        #Gán UE với RU gần nhất
        associations[closest_ru_idx, ue_idx] = 1

        #Kiểm tra xem có đối tượng handover tiềm năng (UE có RU trong khoảng cho phép)
        for ru_idx in range(num_RUs):
            if ru_idx != closest_ru_idx:
                distance = distances_RU_UE[ru_idx, ue_idx]
                distance_diff = distance - min_distance

                #Nếu một RU khác trong khoảng handover của RU gần nhất:
                handover_candidates.append((ue_idx, closest_ru_idx, ru_idx, distance_diff))
    return associations, handover_candidates

def create_and_assign_slices(num_UEs, D_j, D_m, R_min, latencies, bandwidth_requests=None):
    # Nếu không có yêu cầu băng thông, gán slice tốc độ cao nhất
    if bandwidth_requests is None:
        bandwidth_requests = np.full(num_UEs, float('inf'))  # Giả định không có yêu cầu cụ thể

    # Danh sách để lưu thông tin UE
    names, R_min_values, D_j_values, D_m_values, latency_values = [], [], [], [], []

    for bw_request, latency in zip(bandwidth_requests, latencies):
        # Gán slice theo băng thông yêu cầu
        if bw_request < 10:
            assigned_slice = 'eMBB'
        else:
            assigned_slice = 'URLLC'

        # Nếu slice được gán là URLLC, kiểm tra thêm độ trễ
        if assigned_slice == 'URLLC' and latency >= 25:
            assigned_slice = 'eMBB'  # Hạ xuống eMBB nếu độ trễ quá cao

        names.append(assigned_slice)
        R_min_values.append(np.random.choice(R_min) * (2 if assigned_slice == 'URLLC' else 1))
        D_j_values.append(np.random.choice(D_j))
        D_m_values.append(np.random.choice(D_m))
        latency_values.append(latency)  # Lưu giá trị độ trễ

    return names, R_min_values, D_j_values, D_m_values, latency_values

def create_and_assign_slices_with_thresholds(num_UEs, D_j_list, D_m_list, R_min_list, bandwidth_thresholds):
    """
    Assign slices based on bandwidth thresholds and generate requirements
    """
    slice_names = []
    D_j_values = []
    D_m_values = []
    R_min_values = []
    
    # Generate random bandwidth requests for UEs
    bandwidth_requests = np.random.uniform(1, 10, num_UEs)
    
    for bw_request in bandwidth_requests:
        # Assign slice based on bandwidth thresholds
        assigned_slice = "eMBB"  # Default slice
        for slice_type, threshold in bandwidth_thresholds.items():
            if bw_request <= threshold:
                assigned_slice = slice_type
                break
                
        slice_names.append(assigned_slice)
        # Assign requirements based on slice type
        if assigned_slice == "ULLRC":
            R_min_values.append(R_min_list[1])  # Higher requirement for ULLRC
            D_j_values.append(np.random.choice(D_j_list))
            D_m_values.append(np.random.choice(D_m_list))
        else:  # eMBB
            R_min_values.append(R_min_list[0])  # Standard requirement for eMBB
            D_j_values.append(np.random.choice(D_j_list))
            D_m_values.append(np.random.choice(D_m_list))
            
    return slice_names, R_min_values, D_j_values, D_m_values

def plot_save_network(coordinates_RU, coordinates_UE, radius_in, radius_out, 
                     associations=None, handover_candidates=None, handover_threshold=None):

    plt.figure(figsize=(10, 8))
    
    # Vẽ vòng ranh giới
    circle_in = plt.Circle((0, 0), radius_in, color='gray', fill=False, linestyle='--', label='Inner Radius')
    circle_out = plt.Circle((0, 0), radius_out, color='black', fill=False, linestyle='--', label='Outer Radius')
    
    plt.gca().add_artist(circle_in)
    plt.gca().add_artist(circle_out)
    
    # Plot RUs
    for i, (x, y) in enumerate(coordinates_RU):
        label = 'RU' if i == 0 else ""
        plt.scatter(x, y, color='green', marker='^', s=100, label=label)
        plt.annotate(f'RU{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot UEs
    for i, (x, y) in enumerate(coordinates_UE):
        label = 'UE' if i == 0 else ""
        plt.scatter(x, y, color='blue', marker='o', label=label)