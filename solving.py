import numpy as np
import cvxpy as cp
import time
import traceback as tb
from other_function import extract_values
import random
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

def calculate_latency_components(z_ib_sk, R_sk, phi_j_sk, phi_m_sk, c, d_sk, L_cu, L_du,
                                 rho_du, mu_s, lambda_s, constraints):
    """
    Calculate latency components for the optimization problem.
    
    This version fixes DCP rule violations by properly reformulating constraints
    involving division by variables.
    
    Args:
        z_ib_sk: 4D array of binary variables for resource allocation
        R_sk: List of lists containing rate variables for each slice-UE pair
        phi_j_sk, phi_m_sk: Binary variables for DU and CU assignments
        c: Speed of light constant
        d_sk: Distance matrix (nested list or 2D array)
        L_cu, L_du: Processing latencies
        rho_du, mu_s, lambda_s: Service parameters
        constraints: List to append new constraints
    
    Returns:
        Total latency expression
    """
    import cvxpy as cp
    import numpy as np
    
    num_RUs, num_RBs, num_slices, num_UEs = z_ib_sk.shape

    # Improved helper function to safely access nested data structures
    def safe_fetch(data, s, k, default_value=0.0):
        """
        Safely fetch value from potentially nested data structure.
        Handles both list-of-lists and numpy arrays.
        """
        try:
            # First, check if data is a simple list/array with single dimension
            if hasattr(data, '__len__') and not isinstance(data, str):
                if isinstance(data, (list, tuple)):
                    # Handle list of lists
                    if s < len(data):
                        if isinstance(data[s], (list, tuple, np.ndarray)):
                            if k < len(data[s]):
                                return data[s][k]
                            else:
                                # k index out of bounds, use last available
                                return data[s][-1] if len(data[s]) > 0 else default_value
                        else:
                            # data[s] is a scalar, return it
                            return data[s]
                    else:
                        # s index out of bounds, use last available slice
                        last_slice = data[-1] if len(data) > 0 else default_value
                        if isinstance(last_slice, (list, tuple, np.ndarray)):
                            return last_slice[min(k, len(last_slice)-1)] if len(last_slice) > 0 else default_value
                        else:
                            return last_slice
                elif isinstance(data, np.ndarray):
                    # Handle numpy arrays
                    if data.ndim == 2:
                        return data[min(s, data.shape[0]-1), min(k, data.shape[1]-1)]
                    elif data.ndim == 1:
                        return data[min(s, len(data)-1)]
                    else:
                        return default_value
                else:
                    return default_value
            else:
                # Scalar value
                return data
        except Exception as e:
            print(f"Warning in safe_fetch: {e}, using default {default_value}")
            return default_value

    # --- 1. Propagation Latency ---
    print("Calculating propagation latency...")
    L_prop = 0
    try:
        for s in range(num_slices):
            for k in range(num_UEs):
                # Get distance value safely
                d_val = safe_fetch(d_sk, s, k, default_value=0.001)  # Small default to avoid division by zero
                
                # Ensure d_val is a scalar
                if hasattr(d_val, 'value'):  # If it's a CVXPY variable
                    d_val = d_val.value if d_val.value is not None else 0.001
                
                # Sum of allocation variables for this slice-UE pair
                allocation_sum = cp.sum([z_ib_sk[i, b, s, k] for i in range(num_RUs) for b in range(num_RBs)])
                L_prop += (float(d_val) / float(c)) * allocation_sum
    except Exception as e:
        print(f"Error in propagation latency calculation: {e}")
        L_prop = 0

    # --- 2. Transmission Latency (DCP-COMPLIANT VERSION) ---
    print("Calculating transmission latency...")
    L_trans = 0
    trans_components = []
    try:
        for s in range(num_slices):
            for k in range(num_UEs):
                # Get rate variable safely
                R_val = safe_fetch(R_sk, s, k, default_value=None)
                lam = lambda_s[s] if isinstance(lambda_s, (list, tuple)) else lambda_s
                min_rate = 1e-6
                if R_val is None:
                    continue
                elif isinstance(R_val, (cp.Expression, cp.Variable)):
                    constraints.append(R_val >= min_rate)
                    trans_latency_sk = cp.Variable(nonneg=True, name=f"trans_latency_{s}_{k}")
                    constraints.append(trans_latency_sk >= lam * cp.inv_pos(R_val))
                    trans_components.append(trans_latency_sk)
                elif isinstance(R_val, (int, float, np.number)):
                    if R_val > 1e-10:
                        trans_components.append(lam / R_val)
                    else:
                        trans_components.append(1e6)  # Large penalty for very small rates
                else:
                    print(f"Warning: Unexpected R_val type: {type(R_val)}")
                    continue
        if trans_components:
            L_trans = cp.sum(trans_components)
        else:
            L_trans = cp.Constant(0)
    except Exception as e:
        print(f"Error in transmission latency calculation: {e}")
        L_trans = 0

    # --- 3. Queuing Latency (Enhanced DCP Compliance) ---
    print("Calculating queuing latency...")
    L_queue = cp.Variable(nonneg=True, name="queuing_latency")
    queue_components = []
    try:
        for s in range(num_slices):
            # Get service parameters safely
            rho = rho_du[s] if isinstance(rho_du, (list, tuple)) else rho_du
            mu = mu_s[s] if isinstance(mu_s, (list, tuple)) else mu_s
            lam = lambda_s[s] if isinstance(lambda_s, (list, tuple)) else lambda_s
            
            # Enforce stability with larger margin for numerical robustness
            stability_margin = max(1e-3, 0.01 * mu)  # At least 1% of service rate
            constraints.append(mu - lam >= stability_margin)
            
            # Calculate slice load
            slice_load = cp.sum([z_ib_sk[i, b, s, k] 
                               for i in range(num_RUs) 
                               for b in range(num_RBs) 
                               for k in range(num_UEs)])
            
            # For DCP compliance, we can use cp.inv_pos() which is the DCP-compliant
            # way to handle 1/x where x > 0
            # inv_pos(x) is equivalent to 1/x but maintains DCP compliance
            queue_component = rho * cp.inv_pos(mu - lam) * slice_load
            queue_components.append(queue_component)
            
        if queue_components:
            constraints.append(L_queue == cp.sum(queue_components))
        else:
            constraints.append(L_queue == 0)
            
    except Exception as e:
        print(f"Error in queuing latency calculation: {e}")
        L_queue = 0

    # --- 4. Processing Latency ---
    print("Calculating processing latency...")
    L_proc = 0
    try:
        # CU processing latency
        if phi_m_sk.shape[0] > 0:
            cu_processing = cp.sum([phi_m_sk[m, s, k] 
                                  for m in range(phi_m_sk.shape[0])
                                  for s in range(min(num_slices, phi_m_sk.shape[1]))
                                  for k in range(min(num_UEs, phi_m_sk.shape[2]))])
            L_proc += float(L_cu) * cu_processing
        
        # DU processing latency
        if phi_j_sk.shape[0] > 0:
            du_processing = cp.sum([phi_j_sk[j, s, k] 
                                  for j in range(phi_j_sk.shape[0])
                                  for s in range(min(num_slices, phi_j_sk.shape[1]))
                                  for k in range(min(num_UEs, phi_j_sk.shape[2]))])
            L_proc += float(L_du) * du_processing
            
    except Exception as e:
        print(f"Error in processing latency calculation: {e}")
        L_proc = 0

    # Return total latency
    total_latency = L_prop + L_trans + L_queue + L_proc
    print("Latency components calculated successfully")
    return total_latency

def short_term(num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, P_i, gain, R_min, epsilon, 
              arr_pi_sk, arr_phi_i_sk, c=None, d_sk=None, max_latency=None, L_cu=None, L_du=None, 
              rho_du=None, mu_s=None, lambda_s=None, logger=None):
    # Ensure arr_pi_sk and arr_phi_i_sk are 2D arrays of shape (num_slices, num_UEs)
    import numpy as np
    if isinstance(arr_pi_sk, (int, float)):
        arr_pi_sk = np.full((num_slices, num_UEs), arr_pi_sk)
    elif isinstance(arr_pi_sk, (list, tuple, np.ndarray)):
        arr_pi_sk = np.array(arr_pi_sk)
        if arr_pi_sk.ndim == 0:
            arr_pi_sk = np.full((num_slices, num_UEs), arr_pi_sk.item())
        elif arr_pi_sk.ndim == 1:
            arr_pi_sk = np.tile(arr_pi_sk, (num_slices, 1)) if arr_pi_sk.shape[0] == num_UEs else np.tile(arr_pi_sk, (1, num_UEs))
        elif arr_pi_sk.shape != (num_slices, num_UEs):
            arr_pi_sk = np.broadcast_to(arr_pi_sk, (num_slices, num_UEs))
    # Same for arr_phi_i_sk
    if isinstance(arr_phi_i_sk, (int, float)):
        arr_phi_i_sk = np.full((num_RUs, num_slices, num_UEs), arr_phi_i_sk)
    elif isinstance(arr_phi_i_sk, (list, tuple, np.ndarray)):
        arr_phi_i_sk = np.array(arr_phi_i_sk)
        if arr_phi_i_sk.ndim == 0:
            arr_phi_i_sk = np.full((num_RUs, num_slices, num_UEs), arr_phi_i_sk.item())
        elif arr_phi_i_sk.ndim == 1:
            arr_phi_i_sk = np.tile(arr_phi_i_sk, (num_RUs, num_slices, num_UEs))
        elif arr_phi_i_sk.shape != (num_RUs, num_slices, num_UEs):
            arr_phi_i_sk = np.broadcast_to(arr_phi_i_sk, (num_RUs, num_slices, num_UEs))
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

         # Global RB exclusivity - each RB can only be used by ONE (RU, slice, UE) globally
        for b in range(num_RBs):
            constraints.append(cp.sum([short_z_ib_sk[i, b, s, k] for i in range(num_RUs) 
                             for s in range(num_slices) for k in range(num_UEs)]) <= 1)

        # Additional constraint: Each RU-RB pair can only serve one slice-UE combination
        for i in range(num_RUs):
            for b in range(num_RBs):
                constraints.append(cp.sum([short_z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs)]) <= 1)

        # Resource constraint: Each RB index can only be used by one (RU, slice, UE) in the whole network
        for b in range(num_RBs):
            constraints.append(cp.sum([short_z_ib_sk[i, b, s, k] for i in range(num_RUs) for s in range(num_slices) for k in range(num_UEs)]) <= 1)

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

# In the long_term function, modify the R_sk calculation:
def long_term(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, 
              D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping,
              c=None, d_sk=None, max_latency=None, L_cu=None, L_du=None, rho_du=None, mu_s=None, lambda_s=None,
              logger=None):
    try:
        # Initialize decision variables
        # Binary variable: whether RU i allocates RB b to slice s for UE k
        z_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        z_ib_sk[i, b, s, k] = cp.Variable(boolean=True, name=f"z_ib_sk({i}, {b}, {s}, {k})")

        # Continuous variable: base power allocation
        p_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        p_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"p_ib_sk({i}, {b}, {s}, {k})")

        # Continuous variable: actual power allocation (conditional on z_ib_sk)
        mu_ib_sk = np.empty((num_RUs, num_RBs, num_slices, num_UEs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        mu_ib_sk[i, b, s, k] = cp.Variable(nonneg=True, name=f"mu_ib_sk({i}, {b}, {s}, {k})")

        # Binary variables for RU, DU, CU assignments
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

        # Binary variable: whether slice s serves UE k
        pi_sk = cp.Variable((num_slices, num_UEs), boolean=True, name="obj")

        constraints = []
        # --- Use auxiliary rate variables for both latency and constraints ---
        R_sk = []
        for s in range(num_slices):
            row = []
            for k in range(num_UEs):
                # DCP-compliant: sum log(1 + gain * mu) over (i, b)
                rate_expr = cp.sum([
                    rb_bandwidth * cp.log(1 + gain[i, b, s, k] * mu_ib_sk[i, b, s, k]) / np.log(2)
                    for i in range(num_RUs) for b in range(num_RBs)
                ])
                row.append(rate_expr)
            R_sk.append(row)

        # Calculate total data rate (for objective)
        total_R_sk = cp.sum([R_sk[s][k] for s in range(num_slices) for k in range(num_UEs)])
        total_power = cp.sum([mu_ib_sk[i, b, s, k] for i in range(num_RUs) for b in range(num_RBs) for s in range(num_slices) for k in range(num_UEs)])

        # Objective: maximize only the total data rate (with small penalty for power)
        objective = cp.Maximize(total_R_sk * 1e-6 - 1e-9 * total_power)

        # Resource constraints
        for b in range(num_RBs):
            constraints.append(cp.sum([z_ib_sk[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)]) <= 1)
        # QoS constraints: each served UE must meet minimum rate requirement (relaxed)
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(R_sk[s][k] >= 1e-3 * pi_sk[s, k])
        # Power constraints: each RU has limited total power (relaxed)
        for i in range(num_RUs):
            total_power_ru = cp.sum([mu_ib_sk[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
            constraints.append(total_power_ru <= P_i[i])
        # DU and CU resource constraints
        for j in range(num_DUs):
            total_du = cp.sum([phi_j_sk[j, s, k] * D_j[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_du <= A_j[j])
        for m in range(num_CUs):
            total_cu = cp.sum([phi_m_sk[m, s, k] * D_m[k] for s in range(num_slices) for k in range(num_UEs)])
            constraints.append(total_cu <= A_m[m])
        # Mapping constraints: each served UE must be assigned to exactly one RU, DU, and CU
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(cp.sum([phi_i_sk[i, s, k] for i in range(num_RUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_j_sk[j, s, k] for j in range(num_DUs)]) == pi_sk[s, k])
                constraints.append(cp.sum([phi_m_sk[m, s, k] for m in range(num_CUs)]) == pi_sk[s, k])
        # Phi conversion constraints: link RB allocation to RU assignment
        for s in range(num_slices):
            for i in range(num_RUs):
                for k in range(num_UEs):
                    avg_z = (1 / num_RBs) * cp.sum([z_ib_sk[i, b, s, k] for b in range(num_RBs)])
                    constraints.append(avg_z <= phi_i_sk[i, s, k])
                    constraints.append(phi_i_sk[i, s, k] <= avg_z + (1 - epsilon))
        # Connectivity constraints: ensure network topology is respected
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
        # Power allocation constraints: link power variables and enforce limits
        for s in range(num_slices):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    for k in range(num_UEs):
                        max_power_per_allocation = P_i[i] * 0.25
                        constraints.append(mu_ib_sk[i, b, s, k] <= max_power_per_allocation * z_ib_sk[i, b, s, k])
                        constraints.append(mu_ib_sk[i, b, s, k] >= p_ib_sk[i, b, s, k] - max_power_per_allocation * (1 - z_ib_sk[i, b, s, k]))
                        constraints.append(mu_ib_sk[i, b, s, k] <= p_ib_sk[i, b, s, k])
                        normalized_gain = gain[i, b, s, k] / (np.max(gain) + 1e-10)
                        efficient_power = max_power_per_allocation * normalized_gain
                        constraints.append(p_ib_sk[i, b, s, k] <= efficient_power + max_power_per_allocation * (1 - z_ib_sk[i, b, s, k]))
        # Slice mapping constraints: respect predefined slice-UE mappings
        for s in range(num_slices):
            for k in range(num_UEs):
                constraints.append(pi_sk[s, k] == pi_sk[s, k] * slice_mapping[s, k])
        # Add latency constraints if all required parameters are provided
        if all(param is not None for param in [c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s]):
            try:
                total_latency = calculate_latency_components(
                    z_ib_sk, R_sk, phi_j_sk, phi_m_sk, c, d_sk, L_cu, L_du, rho_du, mu_s, lambda_s, constraints
                )
                if total_latency is not None:
                    constraints.append(total_latency <= max_latency)
            except Exception as e:
                if logger:
                    logger.add(f"Warning: Latency constraint calculation failed: {e}")
                else:
                    print(f"Warning: Latency constraint calculation failed: {e}")
        # Solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=SOLVER)

        # Return results based on solver status
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
    
def random_ru_solution(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, 
                       D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping,
                       c=None, d_sk=None, max_latency=None, L_cu=None, L_du=None, rho_du=None, mu_s=None, lambda_s=None,
                       logger=None):
    """
    Random-RU solution implementation following the paper's constraints.
    Ensures: (1) No RB allocation if UE is not selected, (2) Each selected UE gets at least one RB, (3) No RB is assigned to more than one UE.
    """
    try:
        # 1. Random RU assignment (phi_i_sk): for each (s, k), assign one RU only if slice_mapping[s, k] == 1
        phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs))
        for s in range(num_slices):
            for k in range(num_UEs):
                if slice_mapping[s, k] == 1:
                    i = np.random.choice(num_RUs)
                    phi_i_sk[i, s, k] = 1

        # 2. RB assignment: Each selected UE gets at least one unique RB, no RB is assigned to more than one UE
        z_ib_sk = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
        assigned_rbs = set()
        selected_ues = [(s, k) for s in range(num_slices) for k in range(num_UEs) if slice_mapping[s, k] == 1]
        available_rbs = list(range(num_RBs))
        np.random.shuffle(available_rbs)
        # Assign at least one RB to each selected UE
        for idx, (s, k) in enumerate(selected_ues):
            if not available_rbs:
                break  # No more RBs to assign
            b = available_rbs.pop()
            i = np.argmax(phi_i_sk[:, s, k])  # Assign to the RU already selected
            z_ib_sk[i, b, s, k] = 1
            assigned_rbs.add(b)
        # Optionally, assign remaining RBs randomly to selected UEs (but still only one UE per RB)
        for b in range(num_RBs):
            if b in assigned_rbs:
                continue
            if not selected_ues:
                break
            s, k = selected_ues[np.random.randint(len(selected_ues))]
            i = np.argmax(phi_i_sk[:, s, k])
            z_ib_sk[i, b, s, k] = 1
            assigned_rbs.add(b)

        # 3. Power allocation: P_i / |B| for each assigned RB
        p_ib_sk = np.zeros_like(z_ib_sk)
        for i in range(num_RUs):
            for b in range(num_RBs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        if z_ib_sk[i, b, s, k] == 1:
                            p_ib_sk[i, b, s, k] = P_i[i] / num_RBs

        # 4. mu_ib_sk = z * p
        mu_ib_sk = z_ib_sk * p_ib_sk

        # 5. pi_sk: UE is served if it is assigned any RB and slice_mapping[s, k] == 1
        pi_sk = np.zeros((num_slices, num_UEs))
        for s in range(num_slices):
            for k in range(num_UEs):
                if slice_mapping[s, k] == 1 and np.any(z_ib_sk[:, :, s, k] == 1):
                    pi_sk[s, k] = 1

        # 6. Random DU/CU assignment (phi_j_sk, phi_m_sk) for each (s, k) where pi_sk[s, k] == 1
        phi_j_sk = np.zeros((num_DUs, num_slices, num_UEs))
        phi_m_sk = np.zeros((num_CUs, num_slices, num_UEs))
        for s in range(num_slices):
            for k in range(num_UEs):
                if pi_sk[s, k] == 1:
                    j = np.random.choice(num_DUs)
                    m = np.random.choice(num_CUs)
                    phi_j_sk[j, s, k] = 1
                    phi_m_sk[m, s, k] = 1

        # 7. Calculate total_R_sk (rate per (s, k))
        total_R_sk = np.zeros((num_slices, num_UEs))
        for s in range(num_slices):
            for k in range(num_UEs):
                if pi_sk[s, k] == 1:
                    rate = 0
                    for i in range(num_RUs):
                        for b in range(num_RBs):
                            if z_ib_sk[i, b, s, k] == 1:
                                snr = gain[i, b, s, k] * p_ib_sk[i, b, s, k]
                                rate += rb_bandwidth * np.log2(1 + snr)
                    total_R_sk[s, k] = rate

        # 8. Enforce rate constraint: if rate < R_min[k], set pi_sk[s, k] = 0 and zero out assignments
        for s in range(num_slices):
            for k in range(num_UEs):
                if pi_sk[s, k] == 1 and total_R_sk[s, k] < R_min[k]:
                    pi_sk[s, k] = 0
                    z_ib_sk[:, :, s, k] = 0
                    p_ib_sk[:, :, s, k] = 0
                    mu_ib_sk[:, :, s, k] = 0
                    phi_j_sk[:, s, k] = 0
                    phi_m_sk[:, s, k] = 0
                    total_R_sk[s, k] = 0

        # 9. Optionally, check constraints and latency (if required)
        if max_latency is not None and c is not None and d_sk is not None and L_cu is not None and L_du is not None and rho_du is not None and mu_s is not None and lambda_s is not None:
            try:
                # Defensive: ensure d_sk is array of correct shape
                if isinstance(d_sk, (int, float)):
                    d_sk = np.full((num_slices, num_UEs), d_sk)
                elif isinstance(d_sk, (list, tuple)):
                    d_sk = np.array(d_sk)
                    if d_sk.shape != (num_slices, num_UEs):
                        d_sk = np.broadcast_to(d_sk, (num_slices, num_UEs))
                constraints = []
                latency = calculate_latency_components(z_ib_sk, total_R_sk, phi_j_sk, phi_m_sk, c, d_sk, L_cu, L_du, rho_du, mu_s, lambda_s, constraints)
                if latency is not None and hasattr(latency, 'value'):
                    latency_val = latency.value if hasattr(latency, 'value') else latency
                    if logger:
                        logger.add(f"[random_ru] Calculated latency: {latency_val}")
                    if latency_val > max_latency:
                        if logger:
                            logger.add(f"[random_ru] Latency constraint violated: {latency_val} > {max_latency}")
            except Exception as e:
                if logger:
                    logger.add(f"[random_ru] Latency constraint calculation failed: {e}")
                else:
                    print(f"[random_ru] Latency constraint calculation failed: {e}")

        if logger:
            logger.add(f"[random_ru] Random-RU solution completed with total rate: {np.sum(total_R_sk):.2f}")

        return (pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk)

    except Exception as e:
        if logger:
            logger.add(f"[random_ru] ERROR: {e}")
        else:
            print(f'Random-RU error: {e}')
        return None, None, None, None, None, None, None, None
def nearest_ru_solution(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, D_j, D_m, R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping, ue_coords, ru_coords, logger=None):
    """
    Nearest-RU baseline: Assign each UE to its nearest RU, allocate RBs, and distribute power evenly.
    Args:
        ue_coords: shape (num_UEs, 2) array of UE coordinates
        ru_coords: shape (num_RUs, 2) array of RU coordinates
    Returns:
        (pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk)
    """
    try:
        # Ensure coordinates are numpy arrays for vector math
        ue_coords = np.asarray(ue_coords, dtype=float)
        ru_coords = np.asarray(ru_coords, dtype=float)

        # Initialize all arrays
        phi_i_sk = np.zeros((num_RUs, num_slices, num_UEs))
        z_ib_sk = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
        p_ib_sk = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
        mu_ib_sk = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
        pi_sk = np.zeros((num_slices, num_UEs))
        phi_j_sk = np.zeros((num_DUs, num_slices, num_UEs))
        phi_m_sk = np.zeros((num_CUs, num_slices, num_UEs))
        total_R_sk = np.zeros((num_slices, num_UEs))

        # 1. Initial UE-to-RU assignment (only for UEs in slices)
        for s in range(num_slices):
            for k in range(num_UEs):
                if slice_mapping[s, k] == 1:
                    dists = [np.linalg.norm(ue_coords[k] - ru_coords[i]) for i in range(num_RUs)]
                    nearest_i = np.argmin(dists)
                    phi_i_sk[nearest_i, s, k] = 1

        # 2. RB assignment (round-robin among valid UEs)
        for b in range(num_RBs):
            valid_triplets = [(i, s, k) for i in range(num_RUs) for s in range(num_slices) for k in range(num_UEs)
                              if phi_i_sk[i, s, k] == 1 and slice_mapping[s, k] == 1]
            if valid_triplets:
                i, s, k = valid_triplets[b % len(valid_triplets)]
                z_ib_sk[i, b, s, k] = 1

        # 3. Initial power allocation (even distribution per RU)
        for i in range(num_RUs):
            assigned_rbs = np.sum(z_ib_sk[i, :, :, :])
            if assigned_rbs > 0:
                P_i_val = P_i[i] if isinstance(P_i, (list, np.ndarray)) else P_i
                power_per_rb = P_i_val / assigned_rbs
                for b in range(num_RBs):
                    for s in range(num_slices):
                        for k in range(num_UEs):
                            if z_ib_sk[i, b, s, k] == 1:
                                p_ib_sk[i, b, s, k] = power_per_rb

        # 4. mu_ib_sk = z * p
        mu_ib_sk = z_ib_sk * p_ib_sk

        # 5. Calculate rates and determine which UEs can be served
        for s in range(num_slices):
            for k in range(num_UEs):
                if slice_mapping[s, k] == 1 and np.any(z_ib_sk[:, :, s, k]):
                    # Calculate rate for this UE
                    rate = 0
                    for b in range(num_RBs):
                        snr = 0
                        for i in range(num_RUs):
                            snr += gain[i, b, s, k] * mu_ib_sk[i, b, s, k]
                        if snr > 0:
                            rate += rb_bandwidth * np.log2(1 + snr)
                    
                    total_R_sk[s, k] = rate
                    
                    # Check if UE meets minimum rate requirement
                    R_min_val = R_min[k] if isinstance(R_min, (list, np.ndarray)) else R_min
                    if rate >= R_min_val:
                        pi_sk[s, k] = 1
                    else:
                        # UE cannot be served - clear all assignments
                        pi_sk[s, k] = 0
                        phi_i_sk[:, s, k] = 0
                        z_ib_sk[:, :, s, k] = 0
                        p_ib_sk[:, :, s, k] = 0
                        mu_ib_sk[:, :, s, k] = 0
                        total_R_sk[s, k] = 0

        # 6. Recalculate power allocation after removing unservable UEs
        p_ib_sk.fill(0)  # Reset power allocation
        for i in range(num_RUs):
            assigned_rbs = np.sum(z_ib_sk[i, :, :, :])
            if assigned_rbs > 0:
                P_i_val = P_i[i] if isinstance(P_i, (list, np.ndarray)) else P_i
                power_per_rb = P_i_val / assigned_rbs
                for b in range(num_RBs):
                    for s in range(num_slices):
                        for k in range(num_UEs):
                            if z_ib_sk[i, b, s, k] == 1:
                                p_ib_sk[i, b, s, k] = power_per_rb

        # 7. Recalculate mu and rates after power redistribution
        mu_ib_sk = z_ib_sk * p_ib_sk
        
        for s in range(num_slices):
            for k in range(num_UEs):
                if pi_sk[s, k] == 1:
                    rate = 0
                    for b in range(num_RBs):
                        snr = 0
                        for i in range(num_RUs):
                            snr += gain[i, b, s, k] * mu_ib_sk[i, b, s, k]
                        if snr > 0:
                            rate += rb_bandwidth * np.log2(1 + snr)
                    total_R_sk[s, k] = rate

        # 8. DU/CU assignment for served UEs only
        for s in range(num_slices):
            for k in range(num_UEs):
                if pi_sk[s, k] == 1:
                    phi_j_sk[np.random.randint(num_DUs), s, k] = 1
                    phi_m_sk[np.random.randint(num_CUs), s, k] = 1

        if logger:
            served_count = np.sum(pi_sk)
            logger.add(f"[nearest_ru_solution] Completed. Served {served_count} UEs out of {np.sum(slice_mapping)} mapped UEs.")

        return (pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk, total_R_sk)

    except Exception as e:
        if logger:
            logger.add(f"[nearest_ru_solution] ERROR: {e}")
        print(f"Nearest-RU solution error: {e}")
        return None, None, None, None, None, None, None, None