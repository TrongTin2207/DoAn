import numpy as np
import cvxpy as cp
import time
import traceback as tb
from other_function import extract_values
SOLVER = cp.MOSEK

def optimize_power_efficiency(num_slices, num_UEs, num_RUs, num_RBs, P_i, rb_bandwidth, gain, R_min, z_ib_sk, logger=None):
    """Improved power efficiency optimization with proper power distribution"""
    try:
        # Initialize power variables
        p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True)

        # Objective: Minimize total power consumption
        total_power = cp.sum([p_ib_sk[i,b,s,k] for i in range(num_RUs) 
                             for b in range(num_RBs) for s in range(num_slices) for k in range(num_UEs)])
        
        # Secondary objective: Maximize energy efficiency (data rate per unit power)
        total_rate = cp.sum([rb_bandwidth * cp.log(1 + gain[i,b,s,k] * p_ib_sk[i,b,s,k]) / np.log(2) 
                            for i in range(num_RUs) for b in range(num_RBs) 
                            for s in range(num_slices) for k in range(num_UEs)])
        
        # Multi-objective: minimize power while maintaining efficiency
        objective = cp.Minimize(total_power - 1e-6 * total_rate)

        constraints = []

        # Power constraints per RU (use 70% instead of 80% for better distribution)
        for i in range(num_RUs):
            constraints.append(cp.sum([p_ib_sk[i,b,s,k] for b in range(num_RBs) 
                            for s in range(num_slices) for k in range(num_UEs)]) <= 0.7 * P_i[i])

        # QoS constraints - ensure minimum rate requirements are met
        for s in range(num_slices):
            for k in range(num_UEs):
                rate = cp.sum([rb_bandwidth * cp.log(1 + gain[i,b,s,k] * p_ib_sk[i,b,s,k]) / np.log(2)
                             for i in range(num_RUs) for b in range(num_RBs)])
                constraints.append(rate >= R_min[k])

        # Zero power for non-allocated RBs (check if z_ib_sk has values)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        # Use the allocation variable directly if it's already solved
                        if hasattr(z_ib_sk[i,b,s,k], 'value') and z_ib_sk[i,b,s,k].value is not None:
                            if z_ib_sk[i,b,s,k].value < 0.5:
                                constraints.append(p_ib_sk[i,b,s,k] == 0)
                        else:
                            # If not solved yet, add constraint based on allocation variable
                            constraints.append(p_ib_sk[i,b,s,k] <= P_i[i] * z_ib_sk[i,b,s,k])

        # Adaptive power allocation based on channel conditions and fairness
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        # Scale power based on channel gain and available power
                        normalized_gain = gain[i,b,s,k] / (np.max(gain) + 1e-10)
                        max_efficient_power = P_i[i] * normalized_gain * 0.5  # Use 50% for efficiency
                        constraints.append(p_ib_sk[i,b,s,k] <= max_efficient_power)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=SOLVER)

        if problem.status == cp.OPTIMAL:
            # Return the optimized power values
            power_values = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for s in range(num_slices):
                        for k in range(num_UEs):
                            power_values[i,b,s,k] = p_ib_sk[i,b,s,k].value if p_ib_sk[i,b,s,k].value is not None else 0
            return power_values
        
        if logger:
            logger.add(f"[power_opt] Problem status: {problem.status}")
        return None

    except Exception as e:
        if logger:
            logger.add(f"[power_opt] ERROR: {e}")
        print(f"Power optimization error: {e}")
        return None

def short_term(num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, P_i, gain, R_min, epsilon, arr_pi_sk, arr_phi_i_sk, logger=None):
    try:
        # Initialize binary allocation matrix
        short_z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"short_z_ib_sk({i}, {b}, {s}, {k})")

        # Initialize power allocation matrix
        short_p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_p_ib_sk({i}, {b}, {s}, {k})")
        
        # Initialize mu matrix (power allocation considering binary allocation)
        short_mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        short_mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"short_mu_ib_sk({i}, {b}, {s}, {k})")
        
        # Initialize phi variables
        short_phi_i_sk = np.empty((num_RUs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    short_phi_i_sk[i, s, k] = cp.Variable(boolean=True, name=f"short_phi_i_sk({i}, {s}, {k})")
        
        # UE allocation optimization variable
        short_pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="short_pi_sk")

        # Calculate total data rate
        short_total_R_sk = cp.sum([rb_bandwidth * cp.log(1 + cp.sum([gain[i, b, s, k] * short_mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) 
                                  for b in range(num_RBs) for s in range(num_slices) for k in range(num_UEs)])
        
        # Multi-objective: maximize UE allocation and data rate, minimize power
        total_power = cp.sum([short_mu_ib_sk[i, b, s, k] for i in range(num_RUs) for b in range(num_RBs) for s in range(num_slices) for k in range(num_UEs)])
        
        objective = cp.Maximize(cp.sum(short_pi_sk) + 1e-6 * short_total_R_sk - 1e-9 * total_power)

        constraints = []

        # Resource constraint: Only 1 RB per UE per RU
        for b in range(num_RBs):
            constraints.append(cp.sum([short_z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1)

        # Ensure R_min is properly formatted for slices
        if not isinstance(R_min, (list, np.ndarray)) or len(R_min) < num_slices:
            R_min = [R_min] * num_slices  # Use same R_min for all slices if not provided per slice
        
        # QoS constraint with improved power efficiency
        for s in range(num_slices):
            for k in range(num_UEs):
                if arr_pi_sk[s, k] == 1:  # Only apply constraint when UE is allocated
                    R_sk = cp.sum([rb_bandwidth * cp.log(1 + cp.sum([gain[i, b, s, k] * short_mu_ib_sk[i, b, s, k] 
                                 for i in range(num_RUs)])) / np.log(2) for b in range(num_RBs)])
                    constraints.append(R_sk >= R_min[s])

        # Power constraint per RU (use 80% for conservative allocation)
        for i in range(num_RUs):
            total_power_ru = cp.sum([short_mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power_ru <= 0.8 * P_i[i])

        # Improved power-allocation relationship constraints
        for s in range(num_slices):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        # More efficient power allocation
                        max_power_per_allocation = P_i[i] * 0.3  # Limit to 30% per allocation
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= max_power_per_allocation * short_z_ib_sk[i, b, s, k])
                        constraints.append(short_mu_ib_sk[i, b, s, k] >= short_p_ib_sk[i, b, s, k] - max_power_per_allocation * (1 - short_z_ib_sk[i, b, s, k]))
                        constraints.append(short_mu_ib_sk[i, b, s, k] <= short_p_ib_sk[i, b, s, k])
                        
                        # Power efficiency constraint based on channel gain
                        normalized_gain = gain[i,b,s,k] / (np.max(gain) + 1e-10)
                        efficient_power = max_power_per_allocation * normalized_gain
                        constraints.append(short_p_ib_sk[i, b, s, k] <= efficient_power + max_power_per_allocation * (1 - short_z_ib_sk[i, b, s, k]))

        # Phi conversion constraints
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([short_z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= short_phi_i_sk[i, s, k])
                    constraints.append(short_phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Fixed allocation constraints
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    constraints.append(short_phi_i_sk[i, s, k] == arr_phi_i_sk[i, s, k])

        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(short_pi_sk[s, k] == arr_pi_sk[s, k])

        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        if logger is None:
            print(f"{time.process_time()} [solver] actual_solve")
        else:
            logger.add("[solver] actual_solve")
        problem.solve(solver=SOLVER)
        if logger is None:
            print(f"{time.process_time()} [solver] actual_solve {problem.status}")
        else:
            logger.add(f"[solver] actual_solve {problem.status}")

        if problem.status == cp.OPTIMAL:
            return (extract_values(short_pi_sk, int),
                    extract_values(short_z_ib_sk, int),
                    extract_values(short_p_ib_sk, float),
                    extract_values(short_mu_ib_sk, float),
                    short_total_R_sk.value)
        return short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk, short_total_R_sk

    except cp.SolverError as e:
        if logger is None:
            print(f'Solver error: {e}')
        else:
            logger.add(f"[solver] ERROR: {e}")
        return None, None, None, None, None

def long_term(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, logger=None):
    try:
        # Initialize variables (same as before)
        z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"z_ib_sk({i}, {b}, {s}, {k})")

        p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"p_ib_sk({i}, {b}, {s}, {k})")

        mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"mu_ib_sk({i}, {b}, {s}, {k})")

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

        pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="obj")

        # Calculate total data rate and power
        total_R_sk = cp.sum([rb_bandwidth * cp.log(1 + cp.sum([gain[i, b, s, k] * mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) 
                            for b in range(num_RBs) for s in range(num_slices) for k in range(num_UEs)])
        
        total_power = cp.sum([mu_ib_sk[i, b, s, k] for i in range(num_RUs) for b in range(num_RBs) for s in range(num_slices) for k in range(num_UEs)])

        # Multi-objective optimization
        objective = cp.Maximize(gamma * cp.sum(pi_sk) + (1 - gamma) * total_R_sk * 1e-6 - 1e-9 * total_power)

        constraints = []

        # Resource constraints
        for b in range(num_RBs):
            constraints.append(cp.sum([z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1)

        # QoS constraints
        for s in range(num_slices):
            for k in range(num_UEs):
                R_sk = cp.sum([rb_bandwidth * cp.log(1 + cp.sum([gain[i, b, s, k] * mu_ib_sk[i, b, s, k] for i in range(num_RUs)])) / np.log(2) for b in range(num_RBs)])
                constraints.append(R_sk >= R_min * pi_sk[s, k])

        # Improved power constraints
        for i in range(num_RUs):
            total_power_ru = cp.sum([mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power_ru <= 0.75 * P_i[i])  # Use 75% for better efficiency

        # DU and CU resource constraints
        for j in range(num_DUs):
            total_du = cp.sum([phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])

        for m in range(num_CUs):
            total_cu = cp.sum([phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])

        # Mapping constraints
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([phi_i_sk[i, s, k] for i in range(num_RUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_j_sk[j, s, k] for j in range(num_DUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_m_sk[m, s, k] for m in range(num_CUs)]) == pi_sk[s, k])

        # Phi conversion constraints
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= phi_i_sk[i, s, k])
                    constraints.append(phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))

        # Connectivity constraints
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    for j in range(num_DUs):
                        constraints.append(phi_j_sk[j, s, k] <= l_ru_du[i, j] - phi_i_sk[i, s, k] + 1)

        for s in range(num_slices):
            for k in range(num_UEs):
                for j in range(num_DUs):
                    for m in range(num_CUs):
                        constraints.append(phi_m_sk[m, s, k] <= l_du_cu[j, m] - phi_j_sk[j, s, k] + 1)

        # Improved power allocation constraints
        for s in range(num_slices):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        max_power_per_allocation = P_i[i] * 0.25  # Limit to 25% per allocation
                        constraints.append(mu_ib_sk[i, b, s, k] <= max_power_per_allocation * z_ib_sk[i, b, s, k])
                        constraints.append(mu_ib_sk[i, b, s, k] >= p_ib_sk[i, b, s, k] - max_power_per_allocation * (1 - z_ib_sk[i, b, s, k]))
                        constraints.append(mu_ib_sk[i, b, s, k] <= p_ib_sk[i, b, s, k])
                        
                        # Efficient power allocation based on channel conditions
                        normalized_gain = gain[i,b,s,k] / (np.max(gain) + 1e-10)
                        efficient_power = max_power_per_allocation * normalized_gain
                        constraints.append(p_ib_sk[i, b, s, k] <= efficient_power + max_power_per_allocation * (1 - z_ib_sk[i, b, s, k]))

        # Slice mapping constraints
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(pi_sk[s, k] == pi_sk[s, k] * slice_mapping[s, k])

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=SOLVER)

        if problem.status == cp.OPTIMAL:
            return (extract_values(pi_sk, int), 
                extract_values(z_ib_sk, int), 
                extract_values(p_ib_sk, float),  
                extract_values(mu_ib_sk, float),  
                extract_values(phi_i_sk, int),  
                extract_values(phi_j_sk, int),  
                extract_values(phi_m_sk, int), 
                total_R_sk.value)
        
        return pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk

    except cp.SolverError:
        print('Solver error: non_feasible')
        return None, None, None, None, None, None, None, None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None, None, None, None, None, None, None, None