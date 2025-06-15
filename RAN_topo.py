import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list, 
               coordinate_constraints=None, RU_coordinates=None):
    G = nx.Graph()

    # Tạo danh sách các nút RU, DU và CU
    RUs = [f'RU{i+1}' for i in range(num_RUs)]
    DUs = [f'DU{i+1}' for i in range(num_DUs)]
    CUs = [f'CU{i+1}' for i in range(num_CUs)]

    # Thêm các nút RU, DU và CU vào đồ thị
    for i, ru in enumerate(RUs):
        attributes = {
            'type': 'RU',
            'power': np.random.choice(P_j_random_list)
        }
        
        # Add coordinates if provided or generate based on constraints
        if RU_coordinates and i < len(RU_coordinates):
            attributes['x'] = RU_coordinates[i][0]
            attributes['y'] = RU_coordinates[i][1]
        '''
        elif coordinate_constraints:
            attributes['x'] = np.random.uniform(coordinate_constraints['x_min'], coordinate_constraints['x_max'])
            attributes['y'] = np.random.uniform(coordinate_constraints['y_min'], coordinate_constraints['y_max'])
        '''
        G.add_node(ru, **attributes)
        
    for i, du in enumerate(DUs):
        attributes = {
            'type': 'DU',
            'capacity': np.random.choice(A_j_random_list)
        }   
        G.add_node(du, **attributes)
        
    for i, cu in enumerate(CUs):
        attributes = {
            'type': 'CU',
            'capacity': np.random.choice(A_m_random_list)
        }   
        G.add_node(cu, **attributes)

    # Kết nối RUs với DUs (Mỗi DU có thể kết nối với tất cả các RU)
    for du in DUs:
        for ru in RUs:
            G.add_edge(ru, du)

    # Kết nối DUs với CUs (Mỗi DU kết nối với tất cả các CU)
    for du in DUs:
        for cu in CUs:
            G.add_edge(du, cu)

    return G


# Hàm vẽ đồ thị với tọa độ
def draw_topo_coordinates(G):
    """
    Draw the network topology using node coordinates if available, otherwise use default layout
    """
        # Vị trí của các nút: RU, DU, CU xếp thành cột (fallback to default layout)
    RUs = [node for node, data in G.nodes(data=True) if data['type'] == 'RU']
    DUs = [node for node, data in G.nodes(data=True) if data['type'] == 'DU']
    CUs = [node for node, data in G.nodes(data=True) if data['type'] == 'CU']
        
    #Kiểm tra xem tất cả RU có tọa độ chưa
    all_rus_have_coords = all('x' in G.nodes[ru] and 'y' in G.nodes[ru] for ru in RUs)

    #Tạo dictionary chứa vị trí
    pos = {}

    #Vị trí của các RU sử dụng tọa độ của chúng nếu có
    if all_rus_have_coords:
        for ru in RUs:
            x_scale = 0.001
            y_scale = 0.001
            pos[ru] = (G.nodes[ru]['x'] * x_scale, G.nodes[ru]['y'] * y_scale)
        else:
           for i, ru in enumerate(RUs):
               pos[ru] = (-2, 3-i)

        #Sử dụng vị trí được fixed sẵn cho DU và CU (column layout)
        for i, du in enumerate(DUs):
            pos[du] = (0, 2.5 - i*2)
        
        for i, cu in enumerate(CUs):
            pos[cu] = (2, 2-i)

        #Vẽ graph
        plt.figure(figsize=(10,8))

        #Vẽ edge (physical links)
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')

        #Vẽ node với các màu khác nhay
        node_colors = {
            'RU' : 'lightblue',
            'DU' : 'lightgreen',
            'CU' : 'lightcoral'
        }

        #Vẽ mỗi loại khác nhau với nhãn
        for node_type, color in node_colors.items():
            nodes = [node for node, data in G.nodes(data=True) if data['type'] == node_type]

            #Bỏ qua nếu không có nút thuộc dạng này
            if not nodes:
                continue

            nx.draw_networkx_nodes(G, pos,
                                   nodelist=nodes,
                                   node_color=color,
                                   node_size=2000,
                                   linewidths=2)
            #Tạo nhãn dựa trên loại node
            node_labels = {}
            for node in nodes:
                data = G.nodes[node]
                if node_type == 'RU':
                    node_labels[node] = f"{node}\nP: {data['power']}"
                else: 
                    node_labels[node] = f"{node}\nCap: {data['capacity']}"
            nx.draw_networkx_labels(G, pos,
                                    labels=node_labels,
                                    font_size=10,
                                    font_weight='bold',
                                    font_color='black')
            
            if all_rus_have_coords:
                #Lấy bán kính định ước để visualize từ tọa độ RU
                ru_distances = [np.sqrt(G.nodes[ru]['x']**2 + G.nodes[ru]['y']**2) for ru in RUs if ru!= 'RU1']
                if ru_distances:
                    avg_radius = sum(ru_distances) / len(ru_distances)
                    #Vẽ vòng ròn tại 65% bán kính
                    circle = plt.Circle((0, 0), avg_radius * x_scale, fill = False, linestyle = '--', color = 'gray')
                    plt.gca().add_patch(circle)
                
                #Thêm đường reference
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
                plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
            plt.title(f"Network Model: {len(RUs)} RU, {len(DUs)} DU, {len(CUs)} CU", fontsize=15)
            plt.axis('off')
            plt.tight_layout()
            plt.show()


# Original draw function for backward compatibility
def draw_topo(G):
    """
    Draw the network topology using the original column layout
    """
    # Lọc các nút RU, DU và CU từ đồ thị dựa trên thuộc tính 'type'
    RUs = [node for node, data in G.nodes(data=True) if data['type'] == 'RU']
    DUs = [node for node, data in G.nodes(data=True) if data['type'] == 'DU']
    CUs = [node for node, data in G.nodes(data=True) if data['type'] == 'CU']
    # Vị trí của các nút: RU, DU, CU xếp thành cột
    pos = {ru: (0, 3 - i) for i, ru in enumerate(RUs)}
    pos.update({du: (1, 2.5 - i * 2) for i, du in enumerate(DUs)})
    pos.update({cu: (2, 2 - i) for i, cu in enumerate(CUs)})

    # Vẽ đồ thị với các tùy chỉnh
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')

    # Vẽ các nút
    node_colors = ['lightblue' if 'RU' in node else 'lightgreen' if 'DU' in node else 'lightcoral' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, linewidths=2)

    # Hiển thị dung lượng chỉ cho các nút DU và CU
    node_labels = {node: f"{node}\nCap: {data['capacity']}" if 'capacity' in data else f"{node}" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold', font_color='black')

    plt.title(f"Network Model: {len(RUs)} RU, {len(DUs)} DU, {len(CUs)} CU (Column Layout)", fontsize=15)
    plt.axis('off')  # Tắt trục
    plt.tight_layout()  # Điều chỉnh bố cục
    plt.show()


def get_links(G):
    # Lấy danh sách các RU, DU và CU từ đồ thị
    RUs = [node for node, data in G.nodes(data=True) if data['type'] == 'RU']
    DUs = [node for node, data in G.nodes(data=True) if data['type'] == 'DU']
    CUs = [node for node, data in G.nodes(data=True) if data['type'] == 'CU']

    # Khởi tạo ma trận liên kết với tất cả các giá trị ban đầu là 0
    l_ru_du = np.zeros((len(RUs), len(DUs)), dtype=int)  
    l_du_cu = np.zeros((len(DUs), len(CUs)), dtype=int)  

    # Duyệt qua các cạnh để cập nhật ma trận liên kết
    for u, v in G.edges():
        if G.nodes[u]['type'] == 'RU' and G.nodes[v]['type'] == 'DU':
            l_ru_du[RUs.index(u), DUs.index(v)] = 1
        elif G.nodes[u]['type'] == 'DU' and G.nodes[v]['type'] == 'RU':
            l_ru_du[RUs.index(v), DUs.index(u)] = 1
        elif G.nodes[u]['type'] == 'DU' and G.nodes[v]['type'] == 'CU':
            l_du_cu[DUs.index(u), CUs.index(v)] = 1
        elif G.nodes[u]['type'] == 'CU' and G.nodes[v]['type'] == 'DU':
            l_du_cu[DUs.index(v), CUs.index(u)] = 1

    return l_ru_du, l_du_cu


def get_node_cap(G):
    ru_weights = []  # Mảng chứa trọng số của các nút RU
    du_weights = []  # Mảng chứa trọng số của các nút DU
    cu_weights = []  # Mảng chứa trọng số của các nút CU

    # Duyệt qua tất cả các nút trong đồ thị
    for node, data in G.nodes(data=True):
        if data['type'] == 'RU':  # Nếu nút là RU
            ru_weights.append(data['power'])
        if data['type'] == 'DU':  # Nếu nút là DU
            du_weights.append(data['capacity'])
        elif data['type'] == 'CU':  # Nếu nút là CU
            cu_weights.append(data['capacity'])

    return ru_weights, du_weights, cu_weights


# New function to create topology with handover threshold and constraints
def create_topo_with_handover(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list, 
                             handover_threshold, coordinate_constraints=None, RU_coordinates=None):
    """
    Create a network topology with handover threshold and optional coordinate constraints
    
    Parameters:
    - num_RUs, num_DUs, num_CUs: Number of each node type
    - P_j_random_list, A_j_random_list, A_m_random_list: Lists of possible values
    - handover_threshold: Distance threshold for handover
    - coordinate_constraints: Dictionary with x_min, x_max, y_min, y_max to restrict node placement
    - RU_coordinates: Optional pre-defined RU coordinates
    """
    # Create basic topology
    G = create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list, 
                   coordinate_constraints, RU_coordinates)
    
    # Add the handover threshold as a graph attribute
    G.graph['handover_threshold'] = handover_threshold
    
    # Return the graph
    return G


# New function to calculate handover information based on UE positions
def calculate_handover_info(G, UE_coordinates):
    """
    Calculate handover information based on UE positions and RU locations
    
    Parameters:
    - G: Network topology graph
    - UE_coordinates: List of (x, y) coordinates for UEs
    
    Returns:
    - associations: Dictionary mapping UE index to assigned RU node
    - handover_candidates: List of (UE_idx, current_RU, candidate_RU) tuples
    """
    # Get the handover threshold from the graph
    handover_threshold = G.graph.get('handover_threshold', float('inf'))
    
    # Get all RU nodes with their coordinates
    RUs = [(node, data['x'], data['y']) for node, data in G.nodes(data=True) 
           if data['type'] == 'RU' and 'x' in data and 'y' in data]
    
    # If no RUs have coordinates, return empty results
    if not RUs:
        return {}, []
    
    # Calculate associations and handover candidates
    associations = {}
    handover_candidates = []
    
    for ue_idx, (ue_x, ue_y) in enumerate(UE_coordinates):
        # Calculate distances to all RUs
        distances = [(ru_node, np.sqrt((ru_x - ue_x)**2 + (ru_y - ue_y)**2)) 
                     for ru_node, ru_x, ru_y in RUs]
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Associate with closest RU
        closest_ru, min_distance = distances[0]
        associations[ue_idx] = closest_ru
        
        # Check for handover candidates (RUs within threshold of the closest RU)
        for ru_node, distance in distances[1:]:  # Skip the closest one
            if distance - min_distance < handover_threshold:
                handover_candidates.append((ue_idx, closest_ru, ru_node, distance - min_distance))
    
    return associations, handover_candidates
