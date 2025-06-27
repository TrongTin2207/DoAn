import numpy as np

class ValidationLogger:
    def __init__(self):
        self.logs = []

    def add(self, message):
        print(message)
        self.logs.append(message)

    def get_logs(self):
        return self.logs

import numpy as np

def safe_array_access(arr, indices, default=0.0):
    """Safely access array elements with bounds checking"""
    try:
        if isinstance(arr, list):
            arr = arr

        if isinstance(indices, int):
            indices = [indices]

        for i, idx in enumerate(indices):
            if i >= len(arr) or idx >= len(arr[i]):
                return default

        if len(indices) == 1:
            return arr[indices[0]]
        elif len(indices) == 2:
            return arr[indices[0]][indices[1]]
        else:
            raise IndexError("Too many indices")
    except Exception:
        return default

def safe_float(value):
    """Safely convert a value to float, handling None, cvxpy.Variable, numpy arrays, and lists."""
    import numpy as np
    import cvxpy as cp
    # Handle None
    if value is None:
        return 0.0
    # Handle cvxpy objects
    if isinstance(value, (cp.Expression, cp.Variable)):
        try:
            val = value.value
            if val is None:
                return 0.0
            return safe_float(val)
        except Exception:
            return 0.0
    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        if value.shape == () or value.shape == (1,):
            return float(value.item())
        # If it's a 1-element array
        if value.size == 1:
            return float(value.flatten()[0])
        # If it's a multi-element array, take the first element
        return float(np.array(value).flatten()[0])
    # Handle lists or tuples
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        # Recursively extract from the first element
        return safe_float(value[0])
    # Try to convert directly
    try:
        return float(value)
    except Exception:
        return 0.0

def validate_latency_constraints(num_slices, num_UEs, num_RUs, num_RBs, num_DUs, num_CUs,
                                z_ib_sk_val, R_sk_val, phi_j_sk_val, phi_m_sk_val, pi_sk_val,
                                c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s, logger=None):
    """
    Validate latency constraints for uRLLC users with improved stability handling and detailed error reporting.
    Returns (latency_valid, error_list)
    """
    latency_valid = True
    error_list = []
    STABILITY_MARGIN = 1e-4  # Same value as in calculate_latency_components

    d_sk_arr = np.array(d_sk) if isinstance(d_sk, list) else d_sk
    R_sk_arr = np.array(R_sk_val) if isinstance(R_sk_val, list) else R_sk_val
    lambda_s_arr = np.array(lambda_s) if isinstance(lambda_s, list) else lambda_s
    mu_s_arr = np.array(mu_s) if isinstance(mu_s, list) else mu_s
    rho_du_arr = np.array(rho_du) if isinstance(rho_du, list) else rho_du

    if d_sk_arr.ndim == 1:
        d_sk_arr = d_sk_arr.reshape(1, -1)
    if R_sk_arr.ndim == 1:
        R_sk_arr = R_sk_arr.reshape(1, -1)

    for s in range(num_slices):
        for k in range(num_UEs):
            if pi_sk_val[s, k] >= 0.5:  # UE is selected
                # 1. Propagation Latency
                L_prop = 0.0
                for i in range(num_RUs):
                    for b in range(num_RBs):
                        d_val = safe_array_access(d_sk_arr, [s, k], default=1000.0)
                        L_prop += (1.0/c) * d_val/1000.0 * z_ib_sk_val[i, b, s, k]
                # 2. Transmission Latency
                small_constant = 1e-10
                lambda_s_val = safe_array_access(lambda_s_arr, [s], default=1.0)
                R_sk_safe = max(safe_array_access(R_sk_arr, [s, k], default=1e6), small_constant)
                L_trans = lambda_s_val / R_sk_safe
                # 3. Queuing Latency
                L_queue = 0.0
                rho_du_val = safe_array_access(rho_du_arr, [s], default=0.5)
                mu_s_val = safe_array_access(mu_s_arr, [s], default=10.0)
                denominator = mu_s_val - lambda_s_val
                if denominator <= STABILITY_MARGIN:
                    warning_msg = f"Queue stability issue for slice {s}: μ_s ({mu_s_val:.6f}) - Λ_s ({lambda_s_val:.6f}) = {denominator:.6f} <= {STABILITY_MARGIN:.6f}"
                    if logger and hasattr(logger, 'add'):
                        logger.add(f"Warning: {warning_msg}")
                    else:
                        print(f"Warning: {warning_msg}")
                    queue_coefficient = rho_du_val / STABILITY_MARGIN
                    L_queue = queue_coefficient
                    latency_valid = False
                    error_list.append({
                        'type': 'queue_stability',
                        'slice': s, 'ue': k,
                        'mu_s': mu_s_val, 'lambda_s': lambda_s_val,
                        'margin': denominator,
                        'message': warning_msg
                    })
                else:
                    for i in range(num_RUs):
                        for b in range(num_RBs):
                            L_queue += rho_du_val * z_ib_sk_val[i, b, s, k] / denominator
                # 4. Processing Latency
                L_proc = 0.0
                L_cu_val = safe_float(L_cu)
                L_du_val = safe_float(L_du)
                for m in range(num_CUs):
                    L_proc += L_cu_val * phi_m_sk_val[m, s, k]
                for j in range(num_DUs):
                    L_proc += phi_j_sk_val[j, s, k] * L_du_val
                # Total Latency
                total_latency = L_prop + L_trans + L_queue + L_proc
                max_latency_val = safe_float(max_latency)
                tolerance = STABILITY_MARGIN
                # Detailed error reporting for each component
                if L_prop > 0.5 * max_latency_val:
                    error_list.append({
                        'type': 'propagation', 'slice': s, 'ue': k,
                        'value': L_prop, 'threshold': max_latency_val,
                        'message': f'Propagation latency high: {L_prop:.6f} > 0.5*max {max_latency_val:.6f}'
                    })
                if L_trans > 0.5 * max_latency_val:
                    error_list.append({
                        'type': 'transmission', 'slice': s, 'ue': k,
                        'value': L_trans, 'threshold': max_latency_val,
                        'message': f'Transmission latency high: {L_trans:.6f} > 0.5*max {max_latency_val:.6f}'
                    })
                if L_queue > 0.5 * max_latency_val:
                    error_list.append({
                        'type': 'queuing', 'slice': s, 'ue': k,
                        'value': L_queue, 'threshold': max_latency_val,
                        'message': f'Queuing latency high: {L_queue:.6f} > 0.5*max {max_latency_val:.6f}'
                    })
                if L_proc > 0.5 * max_latency_val:
                    error_list.append({
                        'type': 'processing', 'slice': s, 'ue': k,
                        'value': L_proc, 'threshold': max_latency_val,
                        'message': f'Processing latency high: {L_proc:.6f} > 0.5*max {max_latency_val:.6f}'
                    })
                if total_latency > max_latency_val + tolerance:
                    violation_msg = f"Latency violation: UE ({s},{k}) total latency {total_latency:.6f} > max {max_latency_val:.6f} (excess: {total_latency - max_latency_val:.6f})"
                    component_breakdown = [
                        f"  - Propagation: {L_prop:.6f}",
                        f"  - Transmission: {L_trans:.6f}",
                        f"  - Queuing: {L_queue:.6f}",
                        f"  - Processing: {L_proc:.6f}"
                    ]
                    if logger and hasattr(logger, 'add'):
                        logger.add(violation_msg)
                        for line in component_breakdown:
                            logger.add(line)
                    else:
                        print(violation_msg)
                        for line in component_breakdown:
                            print(line)
                    latency_valid = False
                    error_list.append({
                        'type': 'total_latency',
                        'slice': s, 'ue': k,
                        'value': total_latency, 'threshold': max_latency_val,
                        'components': {
                            'propagation': L_prop,
                            'transmission': L_trans,
                            'queuing': L_queue,
                            'processing': L_proc
                        },
                        'message': violation_msg
                    })
                else:
                    success_msg = f"Latency OK: UE ({s},{k}) total latency {total_latency:.6f} <= max {max_latency_val:.6f}"
                    if logger and hasattr(logger, 'debug'):
                        logger.debug(success_msg)
    return latency_valid, error_list

def validate_queue_stability(mu_s, lambda_s, logger=None):
    """
    FIXED: Validate queue stability conditions before optimization
    Returns adjusted parameters that are consistent with optimization constraints
    """
    mu_s_adj = np.array(mu_s) if isinstance(mu_s, list) else mu_s
    lambda_s_adj = np.array(lambda_s) if isinstance(lambda_s, list) else lambda_s
    
    # Use same stability margin as optimization function
    STABILITY_MARGIN = 1e-4
    
    adjustments_made = False
    
    if isinstance(mu_s_adj, np.ndarray) and isinstance(lambda_s_adj, np.ndarray):
        for s in range(len(mu_s_adj)):
            s_idx = min(s, len(lambda_s_adj) - 1)
            
            # FIXED: Use consistent stability margin
            required_margin = STABILITY_MARGIN
            
            if mu_s_adj[s] <= lambda_s_adj[s_idx] + required_margin:
                old_mu = mu_s_adj[s]
                mu_s_adj[s] = lambda_s_adj[s_idx] + required_margin + 1e-6  # Small additional buffer
                adjustments_made = True
                
                msg = f"Adjusted μ_s[{s}] from {old_mu:.6f} to {mu_s_adj[s]:.6f} for stability (margin: {required_margin:.6f})"
                if logger and hasattr(logger, 'add'):
                    logger.add(f"Pre-optimization adjustment: {msg}")
                elif logger:
                    print(f"Pre-optimization adjustment: {msg}")
    
    return mu_s_adj.tolist() if isinstance(mu_s_adj, np.ndarray) else mu_s_adj, adjustments_made

def ensure_optimization_parameter_consistency(mu_s, lambda_s):
    """
    NEW: Ensure parameters are consistent with optimization constraints before solving
    This function should be called before creating the CVXPY problem
    """
    STABILITY_MARGIN = 1e-4
    
    mu_s_arr = np.array(mu_s) if isinstance(mu_s, list) else np.array([mu_s])
    lambda_s_arr = np.array(lambda_s) if isinstance(lambda_s, list) else np.array([lambda_s])
    
    # Ensure mu_s > lambda_s + STABILITY_MARGIN for all slices
    for s in range(len(mu_s_arr)):
        s_idx = min(s, len(lambda_s_arr) - 1)
        if mu_s_arr[s] <= lambda_s_arr[s_idx] + STABILITY_MARGIN:
            print(f"Adjusting μ_s[{s}] for optimization consistency: {mu_s_arr[s]:.6f} -> {lambda_s_arr[s_idx] + STABILITY_MARGIN + 1e-6:.6f}")
            mu_s_arr[s] = lambda_s_arr[s_idx] + STABILITY_MARGIN + 1e-6
    
    return mu_s_arr.tolist() if len(mu_s_arr) > 1 else float(mu_s_arr[0])

def validate_short_term_solution(num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, P_i, gain, R_min, epsilon, 
                                arr_pi_sk, arr_phi_i_sk, pi_sk_result, z_ib_sk_result, p_ib_sk_result, mu_ib_sk_result, 
                                c=None, d_sk=None, max_latency=None, L_cu=None, L_du=None, rho_du=None, mu_s=None, lambda_s=None,
                                logger=None):
    # Ensure arr_pi_sk and arr_phi_i_sk are arrays of the correct shape
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
    
    if logger is None:
        logger = ValidationLogger()
    
    logger.add("\n=== Validating Short Term Solution ===\n")

    # Check if any result is None
    if any(x is None for x in [pi_sk_result, z_ib_sk_result, p_ib_sk_result, mu_ib_sk_result]):
        logger.add("Error: One or more optimization results are None. Validation cannot proceed.")
        return False, None

    # Convert CVXPY variables to numpy arrays for easier validation
    pi_sk_val = np.zeros((num_slices, num_UEs))
    z_ib_sk_val = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
    p_ib_sk_val = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
    mu_ib_sk_val = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))

    # Ensure all values are properly extracted as floats
    for s in range(num_slices):
        for k in range(num_UEs):
            pi_sk_val[s, k] = safe_float(pi_sk_result[s, k])

    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    z_ib_sk_val[i, b, s, k] = safe_float(z_ib_sk_result[i, b, s, k])
                    p_ib_sk_val[i, b, s, k] = safe_float(p_ib_sk_result[i, b, s, k])
                    mu_ib_sk_val[i, b, s, k] = safe_float(mu_ib_sk_result[i, b, s, k])

    # 1. Check RB allocation constraint (each RB index is used by at most one (RU, slice, UE) in the whole network)
    rb_allocation_valid = True
    for b in range(num_RBs):
        total_z = np.sum([z_ib_sk_val[i, b, s, k] for i in range(num_RUs) for s in range(num_slices) for k in range(num_UEs)])
        if total_z > 1 + 1e-6:
            users = []
            for i in range(num_RUs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        if z_ib_sk_val[i, b, s, k] > 0.5:
                            users.append(f"(RU{i},Slice{s},UE{k})")
            logger.add(f"Constraint violation: RB {b} is allocated to more than one UE (sum = {total_z:.4f}). Users: {', '.join(users)}")
            rb_allocation_valid = False
    logger.add(f"RB allocation constraint validated: {rb_allocation_valid}")

    # 2. Check power allocation constraint (total power ≤ P_i)
    power_allocation_valid = True
    for i in range(num_RUs):
        P_i_val = safe_float(P_i[i]) if isinstance(P_i, (list, np.ndarray)) else safe_float(P_i)
        total_power = np.sum([mu_ib_sk_val[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
        if total_power > P_i_val + 1e-6:  # Allow small tolerance
            logger.add(f"Constraint violation: RU {i} exceeds power limit {P_i_val} (used: {total_power:.4f})")
            power_allocation_valid = False
    
    logger.add(f"Power allocation constraint validated: {power_allocation_valid}")
    
    # 3. Check mu = z * p constraint
    mu_constraint_valid = True
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    z_val = z_ib_sk_val[i, b, s, k]
                    p_val = p_ib_sk_val[i, b, s, k]
                    mu_val = mu_ib_sk_val[i, b, s, k]
                    # Use a larger tolerance for numerical errors
                    tol = 1e-2
                    if z_val < 0.5:  # z is 0 (using 0.5 as threshold for binary variables)
                        if abs(mu_val) > tol or abs(p_val) > tol:  # mu and p should be 0
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f}, p = {p_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
                    else:  # z is 1
                        if abs(mu_val - p_val) > tol:  # mu should equal to p
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f} not equal to p = {p_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
    
    logger.add(f"mu = z * p constraint validated: {mu_constraint_valid}")
    
    # 4. Calculate and check data rates (R_sk ≥ R_min * pi_sk) !!! chu y
    rate_constraint_valid = True
    R_sk_val = np.zeros((num_slices, num_UEs))
    
    for s in range(num_slices):
        for k in range(num_UEs):
            logger.add(f"\nCalculating rate for UE({s},{k}):")
            R_sk = 0
            for b in range(num_RBs):
                snr = 0
                for i in range(num_RUs):
                    mu_val = mu_ib_sk_val[i, b, s, k]
                    gain_val = gain[i, b, s, k]
                    contribution = gain_val * mu_val
                    if contribution > 0:
                        logger.add(f"  RU{i} RB{b}: gain={gain_val:.4e}, mu={mu_val:.4e}, contribution={contribution:.4e}")
                    snr += contribution
                    
                if snr > 0:
                    rate_contribution = rb_bandwidth * np.log2(1 + snr)
                    logger.add(f"  RB{b}: SNR={snr:.4e}, rate_contribution={rate_contribution:.4f}")
                    R_sk += rate_contribution
            
            R_sk_val[s, k] = R_sk
            logger.add(f"  Total rate for UE({s},{k}): {R_sk:.4f}")
            
            if pi_sk_val[s, k] > 0:  # UE is selected 
                #nam: sua ve >0 xem co loi frame ko
                # Fix: Safely access R_min values for each slice
                if isinstance(R_min, (list, np.ndarray)):
                    # Ensure we don't go out of bounds
                    if s < len(R_min):
                        R_min_val = safe_float(R_min[s])
                    else:
                        # If the slice index exceeds the R_min list length, use the last value
                        R_min_val = safe_float(R_min[-1])  
                        logger.add(f"Warning: Using fallback R_min value for slice {s}")
                else:
                    R_min_val = safe_float(R_min)
                
                if R_sk < R_min_val - 1e-6:  # Allow small tolerance
                    logger.add(f"Constraint violation: UE ({s},{k}) rate {R_sk:.4f} < slice R_min {R_min_val}")
                    rate_constraint_valid = False
    
    logger.add(f"Data rate constraint validated: {rate_constraint_valid}")
    
    # 5. Check pi_sk and phi_i_sk match input arrays !!! chu y
    pi_sk_match = True
    
    for s in range(num_slices):
        for k in range(num_UEs):
            arr_pi_sk_val = safe_float(arr_pi_sk[s, k])
            if abs(pi_sk_val[s, k] - arr_pi_sk_val) > 1e-6:
                logger.add(f"Constraint violation: pi_sk[{s},{k}] = {pi_sk_val[s,k]:.4f} != arr_pi_sk = {arr_pi_sk_val:.4f}")
                pi_sk_match = False
    
    logger.add(f"pi_sk matches input array: {pi_sk_match}")
    
    # 6. Validate latency constraints if parameters are provided
    latency_constraint_valid = True
    if all(param is not None for param in [c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s]):
        logger.add("Validating latency constraints...")
        # For short term, we assume num_DUs = num_CUs = num_RUs (as placeholders)
        # We need dummy phi arrays for short term validation
        phi_j_sk_val = np.zeros((num_RUs, num_slices, num_UEs))  # Using num_RUs as placeholder
        phi_m_sk_val = np.zeros((num_RUs, num_slices, num_UEs))  # Using num_RUs as placeholder
        
        # For short term, we can assume simple mapping based on phi_i_sk
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    if arr_phi_i_sk[i, s, k] > 0.5:
                        phi_j_sk_val[i, s, k] = 1.0  # Map to corresponding DU
                        phi_m_sk_val[i, s, k] = 1.0  # Map to corresponding CU

        latency_constraint_valid, error_list = validate_latency_constraints(
            num_slices, num_UEs, num_RUs, num_RBs, num_RUs, num_RUs,  # Using num_RUs for DUs and CUs
            z_ib_sk_val, R_sk_val, phi_j_sk_val, phi_m_sk_val, pi_sk_val,
            c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s, logger
        )
    else:
        logger.add("Latency parameters not provided, skipping latency validation")
    
    logger.add(f"Latency constraint validated: {latency_constraint_valid}")

    # 7. Calculate and display objective value (max number of served UEs)
    served_UEs = np.sum(pi_sk_val)
    logger.add(f"Total served UEs: {served_UEs} out of {num_slices * num_UEs}")
    
    # 8. Calculate and display total data rate
    total_rate = np.sum(R_sk_val)
    logger.add(f"Total data rate: {total_rate:.4f}")
    
    # Add validation for power efficiency
    power_efficiency_valid = True
    for i in range(num_RUs):
        total_power = 0
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    total_power += mu_ib_sk_val[i, b, s, k]
        power_efficiency = total_power / P_i[i] if isinstance(P_i, (list, np.ndarray)) else total_power / P_i
        if power_efficiency > 0.9:  # Check if using more than 90% of available power
            logger.add(f"Warning: RU {i} using {power_efficiency*100:.1f}% of available power")
            power_efficiency_valid = False
    
    logger.add(f"Power efficiency validated: {power_efficiency_valid}")

    # Add validation for interference levels
    interference_valid = True
    for b in range(num_RBs):
        for s in range(num_slices):
            for k in range(num_UEs):
                if arr_pi_sk[s, k] > 0:  # Only check active UEs
                    interference = 0
                    signal = 0
                    for i in range(num_RUs):
                        if z_ib_sk_val[i, b, s, k] > 0.5:
                            signal = gain[i, b, s, k] * mu_ib_sk_val[i, b, s, k]
                        else:
                            interference += gain[i, b, s, k] * mu_ib_sk_val[i, b, s, k]
                    if signal > 0 and interference/signal > 0.1:  # Check if interference is more than 10% of signal
                        logger.add(f"Warning: High interference for UE ({s},{k}) on RB {b}: {interference/signal*100:.1f}%")
                        interference_valid = False
    
    logger.add(f"Interference levels validated: {interference_valid}")

    all_valid = (rb_allocation_valid and power_allocation_valid and mu_constraint_valid and 
                rate_constraint_valid and power_efficiency_valid and interference_valid)

    logger.add(f"\nAll constraints validated: {all_valid}")
    
    return all_valid, R_sk_val

def validate_long_term_solution(num_slices, num_UEs, num_RUs, num_DUs, num_CUs, num_RBs, P_i, rb_bandwidth, D_j, D_m, 
                              R_min, gain, A_j, A_m, l_ru_du, l_du_cu, epsilon, gamma, slice_mapping,
                              pi_sk_result, z_ib_sk_result, p_ib_sk_result, mu_ib_sk_result, 
                              phi_i_sk_result, phi_j_sk_result, phi_m_sk_result, 
                              c=None, d_sk=None, max_latency=None, L_cu=None, L_du=None, rho_du=None, mu_s=None, lambda_s=None,
                              logger=None):
    """
    Validate the solution from the long_term optimization model
    """
    
    if logger is None:
        logger = ValidationLogger()
    
    logger.add("\n=== Validating Long Term Solution ===\n")
    
    # Check if any result is None
    if any(x is None for x in [pi_sk_result, z_ib_sk_result, p_ib_sk_result, mu_ib_sk_result, 
                              phi_i_sk_result, phi_j_sk_result, phi_m_sk_result]):
        logger.add("Error: One or more optimization results are None. Validation cannot proceed.")
        return False, None
    
    # Convert CVXPY variables to numpy arrays for easier validation
    pi_sk_val = np.zeros((num_slices, num_UEs))
    z_ib_sk_val = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
    p_ib_sk_val = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
    mu_ib_sk_val = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
    phi_i_sk_val = np.zeros((num_RUs, num_slices, num_UEs))
    phi_j_sk_val = np.zeros((num_DUs, num_slices, num_UEs))
    phi_m_sk_val = np.zeros((num_CUs, num_slices, num_UEs))
    
    # Ensure all values are properly extracted as floats
    for s in range(num_slices):
        for k in range(num_UEs):
            pi_sk_val[s, k] = safe_float(pi_sk_result[s, k])
    
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    z_ib_sk_val[i, b, s, k] = safe_float(z_ib_sk_result[i, b, s, k])
                    p_ib_sk_val[i, b, s, k] = safe_float(p_ib_sk_result[i, b, s, k])
                    mu_ib_sk_val[i, b, s, k] = safe_float(mu_ib_sk_result[i, b, s, k])
    
    for i in range(num_RUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                phi_i_sk_val[i, s, k] = safe_float(phi_i_sk_result[i, s, k])
    
    for j in range(num_DUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                phi_j_sk_val[j, s, k] = safe_float(phi_j_sk_result[j, s, k])
    
    for m in range(num_CUs):
        for s in range(num_slices):
            for k in range(num_UEs):
                phi_m_sk_val[m, s, k] = safe_float(phi_m_sk_result[m, s, k])
    
    # 1. Check RB allocation constraint (each RB index is used by at most one (RU, slice, UE) in the whole network)
    rb_allocation_valid = True
    for b in range(num_RBs):
        total_z = np.sum([z_ib_sk_val[i, b, s, k] for i in range(num_RUs) for s in range(num_slices) for k in range(num_UEs)])
        if total_z > 1 + 1e-6:
            users = []
            for i in range(num_RUs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        if z_ib_sk_val[i, b, s, k] > 0.5:
                            users.append(f"(RU{i},Slice{s},UE{k})")
            logger.add(f"Constraint violation: RB {b} is allocated to more than one UE (sum = {total_z:.4f}). Users: {', '.join(users)}")
            rb_allocation_valid = False
    logger.add(f"RB allocation constraint validated: {rb_allocation_valid}")
    
    # 2. Check power allocation constraint (total power ≤ P_i)
    power_allocation_valid = True
    for i in range(num_RUs):
        P_i_val = safe_float(P_i[i]) if isinstance(P_i, (list, np.ndarray)) else safe_float(P_i)
        total_power = np.sum([mu_ib_sk_val[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
        if total_power > P_i_val + 1e-6:  # Allow small tolerance
            logger.add(f"Constraint violation: RU {i} exceeds power limit {P_i_val} (used: {total_power:.4f})")
            power_allocation_valid = False
    
    logger.add(f"Power allocation constraint validated: {power_allocation_valid}")
    
    # 3. Check mu = z * p constraint
    mu_constraint_valid = True
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    z_val = z_ib_sk_val[i, b, s, k]
                    p_val = p_ib_sk_val[i, b, s, k]
                    mu_val = mu_ib_sk_val[i, b, s, k]
                    # Use a larger tolerance for numerical errors
                    tol = 1e-2
                    if z_val < 0.5:  # z is 0 (using 0.5 as threshold for binary variables)
                        if abs(mu_val) > tol or abs(p_val) > tol:  # mu and p should be 0
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f}, p = {p_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
                    else:  # z is 1
                        if abs(mu_val - p_val) > tol:  # mu should equal to p
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f} not equal to p = {p_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
    
    logger.add(f"mu = z * p constraint validated: {mu_constraint_valid}")
    
    # 4. Calculate and check data rates (R_sk ≥ R_min * pi_sk)
    rate_constraint_valid = True
    R_sk_val = np.zeros((num_slices, num_UEs))
    
    for s in range(num_slices):
        for k in range(num_UEs):
            logger.add(f"\nCalculating rate for UE({s},{k}):")
            R_sk = 0
            for b in range(num_RBs):
                snr = 0
                for i in range(num_RUs):
                    mu_val = mu_ib_sk_val[i, b, s, k]
                    gain_val = gain[i, b, s, k]
                    contribution = gain_val * mu_val
                    if contribution > 0:
                        logger.add(f"  RU{i} RB{b}: gain={gain_val:.4e}, mu={mu_val:.4e}, contribution={contribution:.4e}")
                    snr += contribution
                    
                if snr > 0:
                    rate_contribution = rb_bandwidth * np.log2(1 + snr)
                    logger.add(f"  RB{b}: SNR={snr:.4e}, rate_contribution={rate_contribution:.4f}")
                    R_sk += rate_contribution
            
            R_sk_val[s, k] = R_sk
            logger.add(f"  Total rate for UE({s},{k}): {R_sk:.4f}")
            
            if pi_sk_val[s, k] > 0.5:  # UE is selected
                # Fix: Safely access R_min values for each slice
                if isinstance(R_min, (list, np.ndarray)):
                    # Ensure we don't go out of bounds
                    if s < len(R_min):
                        R_min_val = safe_float(R_min[s])
                    else:
                        # If the slice index exceeds the R_min list length, use the last value
                        R_min_val = safe_float(R_min[-1])
                        logger.add(f"Warning: Using fallback R_min value for slice {s}")
                else:
                    R_min_val = safe_float(R_min)
                
                if R_sk < R_min_val - 1e-6:  # Allow small tolerance
                    logger.add(f"Constraint violation: UE ({s},{k}) rate {R_sk:.4f} < slice R_min {R_min_val}")
                    rate_constraint_valid = False
    
    logger.add(f"Data rate constraint validated: {rate_constraint_valid}")
    
    # 5. Check DU resource constraint (total ≤ A_j)
    du_resource_valid = True
    for j in range(num_DUs):
        A_j_val = safe_float(A_j[j]) if isinstance(A_j, (list, np.ndarray)) else safe_float(A_j)
        total_du = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                D_j_val = safe_float(D_j[k]) if isinstance(D_j, (list, np.ndarray)) else safe_float(D_j)
                total_du += float(phi_j_sk_val[j, s, k]) * D_j_val
                
        if total_du > A_j_val + 1e-6:  # Allow small tolerance
            logger.add(f"Constraint violation: DU {j} exceeds resource limit {A_j_val} (used: {total_du:.4f})")
            du_resource_valid = False
    
    logger.add(f"DU resource constraint validated: {du_resource_valid}")
    
    # 6. Check CU resource constraint (total ≤ A_m)
    cu_resource_valid = True
    for m in range(num_CUs):
        A_m_val = safe_float(A_m[m]) if isinstance(A_m, (list, np.ndarray)) else safe_float(A_m)
        total_cu = 0
        for s in range(num_slices):
            for k in range(num_UEs):
                D_m_val = safe_float(D_m[k]) if isinstance(D_m, (list, np.ndarray)) else safe_float(D_m)
                total_cu += float(phi_m_sk_val[m, s, k]) * D_m_val
                
        if total_cu > A_m_val + 1e-6:  # Allow small tolerance
            logger.add(f"Constraint violation: CU {m} exceeds resource limit {A_m_val} (used: {total_cu:.4f})")
            cu_resource_valid = False
    
    logger.add(f"CU resource constraint validated: {cu_resource_valid}")
    
    # 7. Check that each selected UE is mapped to exactly one RU, DU, and CU
    mapping_valid = True
    for s in range(num_slices):
        for k in range(num_UEs):
            if pi_sk_val[s, k] > 0.5:  # UE is selected
                ru_count = np.sum(phi_i_sk_val[:, s, k])
                du_count = np.sum(phi_j_sk_val[:, s, k])
                cu_count = np.sum(phi_m_sk_val[:, s, k])
                
                if abs(ru_count - 1) > 1e-6:
                    logger.add(f"Constraint violation: UE ({s},{k}) mapped to {ru_count:.1f} RUs (should be 1)")
                    mapping_valid = False
                
                if abs(du_count - 1) > 1e-6:
                    logger.add(f"Constraint violation: UE ({s},{k}) mapped to {du_count:.1f} DUs (should be 1)")
                    mapping_valid = False
                
                if abs(cu_count - 1) > 1e-6:
                    logger.add(f"Constraint violation: UE ({s},{k}) mapped to {cu_count:.1f} CUs (should be 1)")
                    mapping_valid = False
    
    logger.add(f"One-to-one mapping constraint validated: {mapping_valid}")
    
    # 8. Check phi_i_sk vs. z_ib_sk constraint with epsilon
    phi_z_valid = True
    for s in range(num_slices):
        for i in range(num_RUs):
            for k in range(num_UEs):
                # Calculate average z value explicitly with proper type conversion
                sum_z = 0.0
                for b in range(num_RBs):
                    sum_z += safe_float(z_ib_sk_val[i, b, s, k])
                avg_z = sum_z / float(num_RBs) if num_RBs > 0 else 0.0
                
                # Ensure phi_val is a float
                phi_val = safe_float(phi_i_sk_val[i, s, k])
                
                # Now safely perform the comparisons
                if avg_z > phi_val + 1e-6:
                    logger.add(f"Constraint violation: avg_z ({avg_z:.4f}) > phi_i_sk[{i},{s},{k}] ({phi_val:.1f})")
                    phi_z_valid = False
                
                # The problematic operation: make sure (1-epsilon) is converted to float
                epsilon_val = safe_float(epsilon)
                if phi_val > avg_z + float(1.0 - epsilon_val) + 1e-6:
                    logger.add(f"Constraint violation: phi_i_sk[{i},{s},{k}] ({phi_val:.1f}) > avg_z + (1-ε) ({avg_z + (1.0-epsilon_val):.4f})")
                    phi_z_valid = False
    
    logger.add(f"phi_i_sk vs z_ib_sk constraint validated: {phi_z_valid}")
    
    # 9. Check RU-DU connectivity constraint
    ru_du_valid = True
    for s in range(num_slices):
        for k in range(num_UEs):
            for i in range(num_RUs):
                for j in range(num_DUs):
                    if phi_i_sk_val[i, s, k] > 0.5 and phi_j_sk_val[j, s, k] > 0.5:
                        l_ru_du_val = safe_float(l_ru_du[i, j])
                        if l_ru_du_val < 0.5:  # No connection between RU i and DU j
                            logger.add(f"Constraint violation: UE ({s},{k}) uses RU {i} and DU {j} but there's no connection")
                            ru_du_valid = False
    
    logger.add(f"RU-DU connectivity constraint validated: {ru_du_valid}")
    
    # 10. Check DU-CU connectivity constraint
    du_cu_valid = True
    for s in range(num_slices):
        for k in range(num_UEs):
            for j in range(num_DUs):
                for m in range(num_CUs):
                    if phi_j_sk_val[j, s, k] > 0.5 and phi_m_sk_val[m, s, k] > 0.5:
                        l_du_cu_val = safe_float(l_du_cu[j, m])
                        if l_du_cu_val < 0.5:  # No connection between DU j and CU m
                            logger.add(f"Constraint violation: UE ({s},{k}) uses DU {j} and CU {m} but there's no connection")
                            du_cu_valid = False
    
    logger.add(f"DU-CU connectivity constraint validated: {du_cu_valid}")
    
    # 11. Check slice mapping constraint
    slice_map_valid = True
    for s in range(num_slices):
        for k in range(num_UEs):
            slice_mapping_val = safe_float(slice_mapping[s, k])
            if pi_sk_val[s, k] > 0.5 and slice_mapping_val < 0.5:
                logger.add(f"Constraint violation: UE ({s},{k}) is selected but not mapped to slice {s}")
                slice_map_valid = False
    
    logger.add(f"Slice mapping constraint validated: {slice_map_valid}")

    # 12. Validate latency constraints if parameters are provided
    latency_constraint_valid = True
    if all(param is not None for param in [c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s]):
        logger.add("Validating latency constraints...")
        latency_constraint_valid, error_list = validate_latency_constraints(
            num_slices, num_UEs, num_RUs, num_RBs, num_DUs, num_CUs,
            z_ib_sk_val, R_sk_val, phi_j_sk_val, phi_m_sk_val, pi_sk_val,
            c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s, logger
        )
    else:
        logger.add("Latency parameters not provided, skipping latency validation")
    
    logger.add(f"Latency constraint validated: {latency_constraint_valid}")

    # 13. Check eMBB data rate upper bound if applicable
    # NOTE: eMBB should only have a lower bound, not an upper bound. Remove this check.
    embb_rate_valid = True
    # (Removed upper bound check)
    
    # 14. Calculate and display objective value
    served_UEs = np.sum(pi_sk_val)
    total_rate = np.sum(R_sk_val)
    gamma_val = safe_float(gamma)
    obj_value = gamma_val * served_UEs + (1.0 - gamma_val) * total_rate * 1e-6
    
    logger.add(f"Total served UEs: {served_UEs} out of {num_slices * num_UEs}")
    logger.add(f"Total data rate: {total_rate:.4f}")
    logger.add(f"Objective value: {obj_value:.4f}")
    
    all_valid = (rb_allocation_valid and power_allocation_valid and mu_constraint_valid and rate_constraint_valid and
                du_resource_valid and cu_resource_valid and mapping_valid and phi_z_valid and 
                ru_du_valid and du_cu_valid and slice_map_valid and embb_rate_valid) #and latency_constraint_valid
    
    logger.add(f"\nAll constraints validated: {all_valid}")
    
    return all_valid, R_sk_val

def validate_short_term_solution(num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, P_i, gain, R_min, epsilon, 
                                arr_pi_sk, arr_phi_i_sk, pi_sk_result, z_ib_sk_result, p_ib_sk_result, mu_ib_sk_result, 
                                d_sk=None, c=None, max_latency=None, L_cu=None, L_du=None, rho_du=None, mu_s=None, lambda_s=None,
                                logger=None):
    # Ensure arr_pi_sk and arr_phi_i_sk are arrays of the correct shape
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
    
    if logger is None:
        logger = ValidationLogger()
    
    logger.add("\n=== Validating Short Term Solution ===\n")

    # Check if any result is None
    if any(x is None for x in [pi_sk_result, z_ib_sk_result, p_ib_sk_result, mu_ib_sk_result]):
        logger.add("Error: One or more optimization results are None. Validation cannot proceed.")
        return False, None

    # Convert CVXPY variables to numpy arrays for easier validation
    pi_sk_val = np.zeros((num_slices, num_UEs))
    z_ib_sk_val = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
    p_ib_sk_val = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))
    mu_ib_sk_val = np.zeros((num_RUs, num_RBs, num_slices, num_UEs))

    # Ensure all values are properly extracted as floats
    for s in range(num_slices):
        for k in range(num_UEs):
            pi_sk_val[s, k] = safe_float(pi_sk_result[s, k])

    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    z_ib_sk_val[i, b, s, k] = safe_float(z_ib_sk_result[i, b, s, k])
                    p_ib_sk_val[i, b, s, k] = safe_float(p_ib_sk_result[i, b, s, k])
                    mu_ib_sk_val[i, b, s, k] = safe_float(mu_ib_sk_result[i, b, s, k])

    # 1. Check RB allocation constraint (each RB index is used by at most one (RU, slice, UE) in the whole network)
    rb_allocation_valid = True
    for b in range(num_RBs):
        total_z = np.sum([z_ib_sk_val[i, b, s, k] for i in range(num_RUs) for s in range(num_slices) for k in range(num_UEs)])
        if total_z > 1 + 1e-6:
            users = []
            for i in range(num_RUs):
                for s in range(num_slices):
                    for k in range(num_UEs):
                        if z_ib_sk_val[i, b, s, k] > 0.5:
                            users.append(f"(RU{i},Slice{s},UE{k})")
            logger.add(f"Constraint violation: RB {b} is allocated to more than one UE (sum = {total_z:.4f}). Users: {', '.join(users)}")
            rb_allocation_valid = False
    logger.add(f"RB allocation constraint validated: {rb_allocation_valid}")

    # 2. Check power allocation constraint (total power ≤ P_i)
    power_allocation_valid = True
    for i in range(num_RUs):
        P_i_val = safe_float(P_i[i]) if isinstance(P_i, (list, np.ndarray)) else safe_float(P_i)
        total_power = np.sum([mu_ib_sk_val[i, b, s, k] for b in range(num_RBs) for k in range(num_UEs) for s in range(num_slices)])
        if total_power > P_i_val + 1e-6:  # Allow small tolerance
            logger.add(f"Constraint violation: RU {i} exceeds power limit {P_i_val} (used: {total_power:.4f})")
            power_allocation_valid = False
    
    logger.add(f"Power allocation constraint validated: {power_allocation_valid}")
    
    # 3. Check mu = z * p constraint
    mu_constraint_valid = True
    for i in range(num_RUs):
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    z_val = z_ib_sk_val[i, b, s, k]
                    p_val = p_ib_sk_val[i, b, s, k]
                    mu_val = mu_ib_sk_val[i, b, s, k]
                    # Use a larger tolerance for numerical errors
                    tol = 1e-2
                    if z_val < 0.5:  # z is 0 (using 0.5 as threshold for binary variables)
                        if abs(mu_val) > tol or abs(p_val) > tol:  # mu and p should be 0
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f}, p = {p_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
                    else:  # z is 1
                        if abs(mu_val - p_val) > tol:  # mu should equal to p
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f} not equal to p = {p_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
    
    logger.add(f"mu = z * p constraint validated: {mu_constraint_valid}")
    
    # 4. Calculate and check data rates (R_sk ≥ R_min * pi_sk)
    rate_constraint_valid = True
    R_sk_val = np.zeros((num_slices, num_UEs))
    
    for s in range(num_slices):
        for k in range(num_UEs):
            logger.add(f"\nCalculating rate for UE({s},{k}):")
            R_sk = 0
            for b in range(num_RBs):
                snr = 0
                for i in range(num_RUs):
                    mu_val = mu_ib_sk_val[i, b, s, k]
                    gain_val = gain[i, b, s, k]
                    contribution = gain_val * mu_val
                    if contribution > 0:
                        logger.add(f"  RU{i} RB{b}: gain={gain_val:.4e}, mu={mu_val:.4e}, contribution={contribution:.4e}")
                    snr += contribution
                    
                if snr > 0:
                    rate_contribution = rb_bandwidth * np.log2(1 + snr)
                    logger.add(f"  RB{b}: SNR={snr:.4e}, rate_contribution={rate_contribution:.4f}")
                    R_sk += rate_contribution
            
            R_sk_val[s, k] = R_sk
            logger.add(f"  Total rate for UE({s},{k}): {R_sk:.4f}")
            
            if pi_sk_val[s, k] > 0:  # UE is selected 
                #nam: sua ve >0 xem co loi frame ko
                # Fix: Safely access R_min values for each slice
                if isinstance(R_min, (list, np.ndarray)):
                    # Ensure we don't go out of bounds
                    if s < len(R_min):
                        R_min_val = safe_float(R_min[s])
                    else:
                        # If the slice index exceeds the R_min list length, use the last value
                        R_min_val = safe_float(R_min[-1])  
                        logger.add(f"Warning: Using fallback R_min value for slice {s}")
                else:
                    R_min_val = safe_float(R_min)
                
                if R_sk < R_min_val - 1e-6:  # Allow small tolerance
                    logger.add(f"Constraint violation: UE ({s},{k}) rate {R_sk:.4f} < slice R_min {R_min_val}")
                    rate_constraint_valid = False
    
    logger.add(f"Data rate constraint validated: {rate_constraint_valid}")
    
    # 5. Check pi_sk and phi_i_sk match input arrays !!! chu y
    pi_sk_match = True
    
    for s in range(num_slices):
        for k in range(num_UEs):
            arr_pi_sk_val = safe_float(arr_pi_sk[s, k])
            if abs(pi_sk_val[s, k] - arr_pi_sk_val) > 1e-6:
                logger.add(f"Constraint violation: pi_sk[{s},{k}] = {pi_sk_val[s,k]:.4f} != arr_pi_sk = {arr_pi_sk_val:.4f}")
                pi_sk_match = False
    
    logger.add(f"pi_sk matches input array: {pi_sk_match}")
    
    # 6. Validate latency constraints if parameters are provided
    latency_constraint_valid = True
    if all(param is not None for param in [c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s]):
        logger.add("Validating latency constraints...")
        # For short term, we assume num_DUs = num_CUs = num_RUs (as placeholders)
        # We need dummy phi arrays for short term validation
        phi_j_sk_val = np.zeros((num_RUs, num_slices, num_UEs))  # Using num_RUs as placeholder
        phi_m_sk_val = np.zeros((num_RUs, num_slices, num_UEs))  # Using num_RUs as placeholder
        
        # For short term, we can assume simple mapping based on phi_i_sk
        for s in range(num_slices):
            for k in range(num_UEs):
                for i in range(num_RUs):
                    if arr_phi_i_sk[i, s, k] > 0.5:
                        phi_j_sk_val[i, s, k] = 1.0  # Map to corresponding DU
                        phi_m_sk_val[i, s, k] = 1.0  # Map to corresponding CU

        latency_constraint_valid, error_list = validate_latency_constraints(
            num_slices, num_UEs, num_RUs, num_RBs, num_RUs, num_RUs,  # Using num_RUs for DUs and CUs
            z_ib_sk_val, R_sk_val, phi_j_sk_val, phi_m_sk_val, pi_sk_val,
            c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s, logger
        )
    else:
        logger.add("Latency parameters not provided, skipping latency validation")
    
    logger.add(f"Latency constraint validated: {latency_constraint_valid}")

    # 7. Calculate and display objective value (max number of served UEs)
    served_UEs = np.sum(pi_sk_val)
    logger.add(f"Total served UEs: {served_UEs} out of {num_slices * num_UEs}")
    
    # 8. Calculate and display total data rate
    total_rate = np.sum(R_sk_val)
    logger.add(f"Total data rate: {total_rate:.4f}")
    
    # Add validation for power efficiency
    power_efficiency_valid = True
    for i in range(num_RUs):
        total_power = 0
        for b in range(num_RBs):
            for s in range(num_slices):
                for k in range(num_UEs):
                    total_power += mu_ib_sk_val[i, b, s, k]
        power_efficiency = total_power / P_i[i] if isinstance(P_i, (list, np.ndarray)) else total_power / P_i
        if power_efficiency > 0.9:  # Check if using more than 90% of available power
            logger.add(f"Warning: RU {i} using {power_efficiency*100:.1f}% of available power")
            power_efficiency_valid = False
    
    logger.add(f"Power efficiency validated: {power_efficiency_valid}")

    # Add validation for interference levels
    interference_valid = True
    for b in range(num_RBs):
        for s in range(num_slices):
            for k in range(num_UEs):
                if arr_pi_sk[s, k] > 0:  # Only check active UEs
                    interference = 0
                    signal = 0
                    for i in range(num_RUs):
                        if z_ib_sk_val[i, b, s, k] > 0.5:
                            signal = gain[i, b, s, k] * mu_ib_sk_val[i, b, s, k]
                        else:
                            interference += gain[i, b, s, k] * mu_ib_sk_val[i, b, s, k]
                    if signal > 0 and interference/signal > 0.1:  # Check if interference is more than 10% of signal
                        logger.add(f"Warning: High interference for UE ({s},{k}) on RB {b}: {interference/signal*100:.1f}%")
                        interference_valid = False
    
    logger.add(f"Interference levels validated: {interference_valid}")

    all_valid = (rb_allocation_valid and power_allocation_valid and mu_constraint_valid and 
                rate_constraint_valid and power_efficiency_valid and interference_valid)

    logger.add(f"\nAll constraints validated: {all_valid}")
    
    return all_valid, R_sk_val
