import numpy as np
import cvxpy as cp
import time
import traceback as tb

SOLVER = cp.MOSEK

def short_term(num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, P_i, gain, R_min, epsilon,  arr_pi_sk, arr_phi_i_sk, logger=None):
    try:
        # Khởi tạo ma trận nhị phân: short_z_ib_sk (biến xác phân bổ ánh xạ UE k kết nối tới RU i qua RB b tại slice s)
        short_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"short_z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận phân bổ công suất (biến liên tục): short_p_ib_sk (biến xác phân bổ công suất UE k kết nối tới RU i qua RB b tại slice s)
        short_p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_p_ib_sk({i}, {b}, {s}, {k})")
        
        # Khởi tạo ma trận mu (biến liên tục): short_mu_ib_sk = short_z_ib_sk * short_P_ib_sk
        short_mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_mu_ib_sk({i}, {b}, {s}, {k})")
        
        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        short_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    short_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"short_phi_i_sk({i}, {s}, {k})")
        
        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        short_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    short_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"short_phi_i_sk({i}, {s}, {k})")
        
        # Biến tối ưu cho việc phân bổ
        short_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="short_pi_sk")  # Tối ưu số lượng UE

        short_total_R_sk = 0
        # Hàm mục tiêu: Tối ưu tổng công suất phân bổ
        #objective = cp.Maximize(cp.sum([short_mu_ib_sk[i, b, s, k] for i in range(num_RUs) for b in range(num_RBs) for s in range(num_slices) for k in range(num_UEs)]))
        objective = cp.Maximize(cp.sum(short_pi_sk))

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([short_z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1)
        

        # Ràng buộc: Đảm bảo QoS (Data rate)
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([rb_bandwidth * cp.log(1 + cp.sum([gain[i, b, s, k] * short_mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) for b in range(num_RBs)])
                constraints.append(R_sk >= R_min * short_pi_sk[s, k])
                short_total_R_sk += R_sk
                

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([short_mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power <= P_i[i])

        # Ràng buộc bổ sung: Chuyển đổi tích sang tổng (mu_ib_sk = z_ib_sk * p_ib_sk)
        for s in range(num_slices):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= P_i[i] * short_z_ib_sk[i, b, s, k])  
                        constraints.append(short_mu_ib_sk[i, b, s, k] >= short_p_ib_sk[i, b, s, k] - P_i[i] * (1 - short_z_ib_sk[i, b, s, k]))
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= short_p_ib_sk[i, b, s, k])  

        # Ràng buộc: Chuyển đổi short_z_ib_sk sang short_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([short_z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= short_phi_i_sk[i, s, k])
                    constraints.append(short_phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Chuyển đổi short_phi_i_sk == arr_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    constraints.append(short_phi_i_sk[i, s, k] == arr_phi_i_sk[i, s, k])

        # Ràng buộc: Chuyển đổi short_pi_sk == arr_pi_sk
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(short_pi_sk[s, k] == arr_pi_sk[s, k])

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        if logger is None:
            print(f"{time.process_time()} [solver] actual_solve")
        else:
            logger.add("[solver] actual_solve")
        problem.solve(solver = SOLVER)
        if logger is None:
            print(f"{time.process_time()} [solver] actual_solve {problem.status}")
        else:
            logger.add(f"[solver] actual_solve {problem.status}")

        # Trả về kết quả tối ưu công suất và ánh xạ RB
        return short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk, short_total_R_sk

    except cp.SolverError as e:
        if logger is None:
            print(f'Solver error: {e}')
        else:
            logger.add(f"[solver] ERROR: {e}")
        return None, None, None, None, None


def long_term(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, logger = None):
    try:
        # Khởi tạo ma trận nhị phân: z_ib_sk
        z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận phân bổ công suất (biến liên tục): p_ib_sk
        p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"p_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận mu (biến liên tục): mu_ib_sk = z_ib_sk * p_ib_sk
        mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"mu_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"phi_i_sk({i}, {s}, {k})")

        phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=object)
        for j in range(num_DUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_j_sk[j, s, k] = cp.Variable(boolean=True, name=f"phi_j_sk({j}, {s}, {k})")

        phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=object)
        for m in range(num_CUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_m_sk[m, s, k] = cp.Variable(boolean=True, name=f"phi_m_sk({m}, {s}, {k})")
        # Biến tối ưu cho việc phân bổ
        pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="obj") 

        total_R_sk = 0

        objective = cp.Maximize(gamma * cp.sum(pi_sk) + (1 - gamma) * total_R_sk * 1e-6)

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1) 

        # Ràng buộc: Đảm bảo QoS (Data rate)
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([rb_bandwidth * cp.log(1 + cp.sum([gain[i, b, s, k] * mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) for b in range(num_RBs)])

                constraints.append(R_sk  >= R_min * pi_sk[s, k])
                total_R_sk += R_sk

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power <= P_i[i])

        # Ràng buộc: Tổng tài nguyên của slice sử dụng DU <= tài nguyên có sẵn tại DU
        for j in range(num_DUs):
            total_du = cp.sum([phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])

        # Ràng buộc: Tổng tài nguyên của slice sử dụng CU <= tài nguyên có sẵn tại CU
        for m in range(num_CUs):
            total_cu = cp.sum([phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # Ràng buộc: Đảm bảo ánh xạ toàn bộ RU, DU, CU
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([phi_i_sk[i, s, k] for i in range(num_RUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_j_sk[j, s, k] for j in range(num_DUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_m_sk[m, s, k] for m in range(num_CUs)]) == pi_sk[s, k])

        # Ràng buộc: Chuyển đổi z_ib_sk sang phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= phi_i_sk[i, s, k])
                    constraints.append(phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Đảm bảo liên kết từ RU - DU
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(phi_j_sk[j, s, k] <= l_ru_du[i, j] - phi_i_sk[i, s, k] + 1)

        # Ràng buộc: Đảm bảo liên kết từ DU - CU
        for s in range(num_slices):
            for k in range(num_UEs):
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(phi_m_sk[m, s, k] <= l_du_cu[j, m] - phi_j_sk[j, s, k] + 1)

        # Ràng buộc bổ sung: Chuyển đổi tích sang tổng (mu_ib_sk = z_ib_sk * p_ib_sk)
        for s in range(num_slices):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(mu_ib_sk[i, b, s, k] <= P_i[i] * z_ib_sk[i, b, s, k])  
                        constraints.append(mu_ib_sk[i, b, s, k] >= p_ib_sk[i, b, s, k] - P_i[i] * (1 - z_ib_sk[i, b, s, k]))
                        constraints.append(mu_ib_sk[i, b, s, k] <= p_ib_sk[i, b, s, k])  

        # Đảm bảo chỉ có các UE ở các slice được ánh xạ
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(pi_sk[s, k] == pi_sk[s, k] * slice_mapping[s, k])

        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        problem.solve(solver = cp.MOSEK)

        #with open('./Wireless-RAN-Mapping/result/demo-debug.prob', 'wt') as f:
        #    f.write(str(problem))
        
        return pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None, None

def mapping_RU_nearest_UE(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, arr_phi_i_sk, logger = None):
    try:
        # Khởi tạo ma trận nhị phân: nearest_z_ib_sk
        nearest_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        nearest_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"nearest_z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận phân bổ công suất (biến liên tục): nearest_p_ib_sk
        nearest_p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        nearest_p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"nearest_p_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận mu (biến liên tục): nearest_mu_ib_sk = nearest_z_ib_sk * p_ib_sk
        nearest_mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        nearest_mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"nearest_mu_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo các biến nhị phân: nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk
        nearest_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    nearest_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"nearest_phi_i_sk({i}, {s}, {k})")

        nearest_phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=object)
        for j in range(num_DUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    nearest_phi_j_sk[j, s, k] = cp.Variable(boolean=True, name=f"nearest_phi_j_sk({j}, {s}, {k})")

        nearest_phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=object)
        for m in range(num_CUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    nearest_phi_m_sk[m, s, k] = cp.Variable(boolean=True, name=f"nearest_phi_m_sk({m}, {s}, {k})")

        # Biến tối ưu cho việc phân bổ
        nearest_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="obj") 

        nearest_total_R_sk = 0

        objective = cp.Maximize(gamma * cp.sum(nearest_pi_sk) + (1 - gamma) * nearest_total_R_sk * 1e-6)

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([nearest_z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1)

        # Ràng buộc: Đảm bảo QoS (Data rate)
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([rb_bandwidth * cp.log(1 + cp.sum([gain[i, b, s, k] * nearest_mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) for b in range(num_RBs)])

                constraints.append(R_sk >= R_min * nearest_pi_sk[s, k])
                nearest_total_R_sk += R_sk

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([nearest_mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power <= P_i[i])

        # Ràng buộc: Tổng tài nguyên của slice sử dụng DU <= tài nguyên có sẵn tại DU
        for j in range(num_DUs):
            total_du = cp.sum([nearest_phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])
        

        # Ràng buộc: Tổng tài nguyên của slice sử dụng CU <= tài nguyên có sẵn tại CU
        for m in range(num_CUs):
            total_cu = cp.sum([nearest_phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # Ràng buộc: Đảm bảo ánh xạ toàn bộ RU, DU, CU
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([nearest_phi_i_sk[i, s, k] for i in range(num_RUs)]) == nearest_pi_sk[s, k])
                constraints.append(cp.sum([nearest_phi_j_sk[j, s, k] for j in range(num_DUs)]) == nearest_pi_sk[s, k])
                constraints.append(cp.sum([nearest_phi_m_sk[m, s, k] for m in range(num_CUs)]) == nearest_pi_sk[s, k])

        # Ràng buộc: Chuyển đổi z_ib_sk sang phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([nearest_z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= nearest_phi_i_sk[i, s, k])
                    constraints.append(nearest_phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Đảm bảo liên kết từ RU - DU
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(nearest_phi_j_sk[j, s, k] <= l_ru_du[i, j] - nearest_phi_i_sk[i, s, k] + 1)

        # Ràng buộc: Đảm bảo liên kết từ DU - CU
        for s in range(num_slices):
            for k in range(num_UEs):
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(nearest_phi_m_sk[m, s, k] <= l_du_cu[j, m] - nearest_phi_j_sk[j, s, k] + 1)

        # Ràng buộc bổ sung: Chuyển đổi tích sang tổng (mu_ib_sk = z_ib_sk * p_ib_sk)
        for s in range(num_slices):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(nearest_mu_ib_sk[i, b, s, k] <= P_i[i] * nearest_z_ib_sk[i, b, s, k])  
                        constraints.append(nearest_mu_ib_sk[i, b, s, k] >= nearest_p_ib_sk[i, b, s, k] - P_i[i] * (1 - nearest_z_ib_sk[i, b, s, k]))
                        constraints.append(nearest_mu_ib_sk[i, b, s, k] <= nearest_p_ib_sk[i, b, s, k])  
        
        # Đảm bảo chỉ có các UE ở các slice được ánh xạ
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(nearest_pi_sk[s, k] == nearest_pi_sk[s, k] * slice_mapping[s, k])

        # Ràng buộc: phi_i_sk == nearest_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    constraints.append(nearest_phi_i_sk[i, s, k] == nearest_phi_i_sk[i, s, k] * arr_phi_i_sk[i, s, k])

   
        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        if logger is None:
            print(f"{time.process_time()} [solver] actual_solve")
        else:
            logger.add("[solver] actual_solve")
        problem.solve(solver = SOLVER)
        if logger is None:
            print(f"{time.process_time()} [solver] actual_solve {problem.status}")
        else:
            logger.add(f"[solver] actual_solve {problem.status}")

        #with open('./Wireless-RAN-Mapping/result/demo-debug.prob', 'wt') as f:
        #    f.write(str(problem))
        
        
        return nearest_pi_sk, nearest_z_ib_sk, nearest_p_ib_sk, nearest_mu_ib_sk, nearest_phi_i_sk, nearest_phi_j_sk, nearest_phi_m_sk, nearest_total_R_sk

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None

def mapping_equal_power(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, nearest_phi_i_sk):
    try:
        # Khởi tạo ma trận nhị phân: z_ib_sk
        z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"z_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận phân bổ công suất (biến liên tục): p_ib_sk
        p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"p_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo ma trận mu (biến liên tục): mu_ib_sk = z_ib_sk * p_ib_sk
        mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"mu_ib_sk({i}, {b}, {s}, {k})")

        # Khởi tạo các biến nhị phân: phi_i_sk, phi_j_sk, phi_m_sk
        phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"phi_i_sk({i}, {s}, {k})")

        phi_j_sk = np.empty((num_DUs, num_slices, num_UEs), dtype=object)
        for j in range(num_DUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_j_sk[j, s, k] = cp.Variable(boolean=True, name=f"phi_j_sk({j}, {s}, {k})")

        phi_m_sk = np.empty((num_CUs, num_slices, num_UEs), dtype=object)
        for m in range(num_CUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    phi_m_sk[m, s, k] = cp.Variable(boolean=True, name=f"phi_m_sk({m}, {s}, {k})")
        # Biến tối ưu cho việc phân bổ
        pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="obj") 

        total_data_rate = cp.sum([rb_bandwidth * cp.log(1 + cp.sum([gain[i, b, s, k] * mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) for s in range(num_slices) for k in range(num_UEs) for b in range(num_RBs)]) 

        objective = cp.Maximize(gamma * cp.sum(pi_sk) + (1 - gamma) * total_data_rate * 1e-6)

        # Danh sách ràng buộc
        constraints = []

        # Ràng buộc: Chỉ 1 RB được sử dụng cho 1 UE tại 1 RU
        for b in range(num_RBs):
            constraints.append(cp.sum([z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1)

        # Ràng buộc: Đảm bảo QoS (Data rate)
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([rb_bandwidth * cp.log(1 + cp.sum([gain[i, b, s, k] * mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) for b in range(num_RBs)])

                constraints.append(R_sk >= R_min * pi_sk[s, k])

        # Ràng buộc: Tổng công suất phân bổ <= công suất tối đa của RU
        for i in range(num_RUs):
            total_power = cp.sum([mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power <= P_i[i])

        # Ràng buộc: Tổng tài nguyên của slice sử dụng DU <= tài nguyên có sẵn tại DU
        for j in range(num_DUs):
            total_du = cp.sum([phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])

        # Ràng buộc: Tổng tài nguyên của slice sử dụng CU <= tài nguyên có sẵn tại CU
        for m in range(num_CUs):
            total_cu = cp.sum([phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # Ràng buộc: Đảm bảo ánh xạ toàn bộ RU, DU, CU
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([phi_i_sk[i, s, k] for i in range(num_RUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_j_sk[j, s, k] for j in range(num_DUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_m_sk[m, s, k] for m in range(num_CUs)]) == pi_sk[s, k])

        # Ràng buộc: Chuyển đổi z_ib_sk sang phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= phi_i_sk[i, s, k])
                    constraints.append(phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Ràng buộc: Đảm bảo liên kết từ RU - DU
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(phi_j_sk[j, s, k] <= l_ru_du[i, j] - phi_i_sk[i, s, k] + 1)

        # Ràng buộc: Đảm bảo liên kết từ DU - CU
        for s in range(num_slices):
            for k in range(num_UEs):
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(phi_m_sk[m, s, k] <= l_du_cu[j, m] - phi_j_sk[j, s, k] + 1)

        # Ràng buộc bổ sung: Chuyển đổi tích sang tổng (mu_ib_sk = z_ib_sk * p_ib_sk)
        for s in range(num_slices):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        constraints.append(mu_ib_sk[i, b, s, k] <= P_i[i] * z_ib_sk[i, b, s, k])  
                        constraints.append(mu_ib_sk[i, b, s, k] >= p_ib_sk[i, b, s, k] - P_i[i] * (1 - z_ib_sk[i, b, s, k]))
                        constraints.append(mu_ib_sk[i, b, s, k] <= p_ib_sk[i, b, s, k])  
        
        # Đảm bảo chỉ có các UE ở các slice được ánh xạ
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(pi_sk[s, k] == pi_sk[s, k] * slice_mapping[s, k])

        # Ràng buộc: phi_i_sk == nearest_phi_i_sk
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    constraints.append(phi_i_sk[i, s, k] == phi_i_sk[i, s, k] * nearest_phi_i_sk[i, s, k])

   
        # Giải bài toán tối ưu
        problem = cp.Problem(objective, constraints)
        problem.solve(solver = cp.MOSEK)

        #with open('./Wireless-RAN-Mapping/result/demo-debug.prob', 'wt') as f:
        #    f.write(str(problem))
        
        return pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_data_rate

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None, None
