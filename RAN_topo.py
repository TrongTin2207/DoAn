import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def create_topo(num_RUs, num_DUs, num_CUs, P_j_random_list, A_j_random_list, A_m_random_list):
    G = nx.Graph()

    # Tạo danh sách các nút RU, DU và CU
    RUs = [f'RU{i+1}' for i in range(num_RUs)]
    DUs = [f'DU{i+1}' for i in range(num_DUs)]
    CUs = [f'CU{i+1}' for i in range(num_CUs)]

    # Thêm các nút RU, DU và CU vào đồ thị
    for ru in RUs:
        G.add_node(ru, type='RU', power = np.random.choice(P_j_random_list))
    for du in DUs:
        G.add_node(du, type='DU', capacity = np.random.choice(A_j_random_list))
    for cu in CUs:
        G.add_node(cu, type='CU', capacity = np.random.choice(A_m_random_list))

    # Kết nối RUs với DUs (Mỗi DU có thể kết nối với tất cả các RU)
    for du in DUs:
        for ru in RUs:
            G.add_edge(ru, du)

    # Kết nối DUs với CUs (Mỗi DU kết nối với tất cả các CU)
    for du in DUs:
        for cu in CUs:
            G.add_edge(du, cu)

    return G


# Hàm vẽ đồ thị
def draw_topo(G):
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
