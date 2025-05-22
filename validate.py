import numpy as np

class ValidationLogger:
    def __init__(self):
        self.logs = []

    def add(self, message):
        print(message)
        self.logs.append(message)

    def get_logs(self):
        return self.logs
    
def safe_float(value):
    """Safely convert a value to float, handling None values and CVXPY variables."""
    if value is None:
        return 0.0
    if hasattr(value, 'value'):
        if value.value is None:
            return 0.0
        return float(value.value)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return 0.0  # Default case

def validate_latency_constraints(num_slices, num_UEs, num_RUs, num_RBs, num_DUs, num_CUs,
                                z_ib_sk_val, R_sk_val, phi_j_sk_val, phi_m_sk_val, pi_sk_val,
                                c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s, logger):
    """
    Validate latency constraints for uRLLC users
    """
    latency_valid = True
    
    for s in range(num_slices):
        for k in range(num_UEs):
            if pi_sk_val[s, k] > 0.5:  # UE is selected
                # Calculate latency components
                
                # 1. Propagation Latency: L_s,k^prop(t) = (1/c) * d_s,k * z_s,k^bi[t]
                L_prop = 0.0
                for i in range(num_RUs):
                    for b in range(num_RBs):
                        L_prop += (1.0/c) * safe_float(d_sk[s, k]) * z_ib_sk_val[i, b, s, k]
                
                # 2. Transmission Latency: L_s,k^trans = Λ_s / R_s,k
                small_constant = 1e-10
                lambda_s_val = safe_float(lambda_s[s])
                R_sk_safe = max(R_sk_val[s, k], small_constant)  # Avoid division by zero
                L_trans = lambda_s_val / R_sk_safe
                
                # 3. Queuing Latency: L_s,k^queue = ρ_du * z_s,k^bi / (μ_s - Λ_s)
                L_queue = 0.0
                rho_du_val = safe_float(rho_du[s])
                mu_s_val = safe_float(mu_s[s])
                denominator = mu_s_val - lambda_s_val
                
                if denominator > 1e-10:  # Avoid division by zero
                    for i in range(num_RUs):
                        for b in range(num_RBs):
                            L_queue += rho_du_val * z_ib_sk_val[i, b, s, k] / denominator
                else:
                    logger.add(f"Warning: μ_s - Λ_s is too small for slice {s}, queuing latency may be invalid")
                    L_queue = float('inf')  # Invalid queuing latency
                
                # 4. Processing Latency: L_s,k^proc = L_cu * ϕ_s,k^m + ϕ_s,k^j * L_du
                L_proc = 0.0
                L_cu_val = safe_float(L_cu)
                L_du_val = safe_float(L_du)
                
                # CU processing latency
                for m in range(num_CUs):
                    L_proc += L_cu_val * phi_m_sk_val[m, s, k]
                
                # DU processing latency
                for j in range(num_DUs):
                    L_proc += phi_j_sk_val[j, s, k] * L_du_val
                
                # Total Latency
                total_latency = L_prop + L_trans + L_queue + L_proc
                max_latency_val = safe_float(max_latency)
                
                # Validate latency constraint
                if total_latency > max_latency_val + 1e-6:  # Allow small tolerance
                    logger.add(f"Latency violation: UE ({s},{k}) total latency {total_latency:.6f} > max {max_latency_val}")
                    logger.add(f"  - Propagation: {L_prop:.6f}")
                    logger.add(f"  - Transmission: {L_trans:.6f}")
                    logger.add(f"  - Queuing: {L_queue:.6f}")
                    logger.add(f"  - Processing: {L_proc:.6f}")
                    latency_valid = False
    
    return latency_valid

def validate_short_term_solution(num_slices, num_UEs, num_RUs, num_RBs, rb_bandwidth, P_i, gain, R_min, epsilon, 
                                arr_pi_sk, arr_phi_i_sk, pi_sk_result, z_ib_sk_result, p_ib_sk_result, mu_ib_sk_result, 
                                c=None, d_sk=None, max_latency=None, L_cu=None, L_du=None, rho_du=None, mu_s=None, lambda_s=None,
                                logger=None):
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

    # 1. Check RB allocation constraint (each RB is used by at most one UE)
    rb_allocation_valid = True
    for b in range(num_RBs):
        total_z = np.sum([z_ib_sk_val[i,b,s,k] for s in range(num_slices)
                          for k in range(num_UEs)
                          for i in range(num_RUs)])
        if total_z > 1 + 1e-6:
            logger.add(f"Constraint violation: RB {b} is allocated to more than one UE (sum = {total_z:.4f})")
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
                    
                    if z_val < 0.5:  # z is 0 (using 0.5 as threshold for binary variables)
                        if mu_val > 1e-6:  # mu should be 0
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
                    else:  # z is 1
                        if abs(mu_val - p_val) > 1e-6:  # mu should equal p
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f} not equal to p = {p_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
    
    logger.add(f"mu = z * p constraint validated: {mu_constraint_valid}")
    
    # 4. Calculate and check data rates (R_sk ≥ R_min * pi_sk)
    rate_constraint_valid = True
    R_sk_val = np.zeros((num_slices, num_UEs))
    
    for s in range(num_slices):
        for k in range(num_UEs):
            R_sk = 0
            for b in range(num_RBs):
                snr = 0
                for i in range(num_RUs):
                    snr += gain[i, b, s, k] * mu_ib_sk_val[i, b, s, k]
                if snr > 0:
                    R_sk += rb_bandwidth * np.log(1 + snr) / np.log(2)
            
            R_sk_val[s, k] = R_sk
            
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
    
    # 5. Check pi_sk and phi_i_sk match input arrays
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
        
        latency_constraint_valid = validate_latency_constraints(
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
    
    all_valid = rb_allocation_valid and power_allocation_valid and mu_constraint_valid and rate_constraint_valid and pi_sk_match and latency_constraint_valid
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
    
    # 1. Check RB allocation constraint (each RB is used by at most one UE)
    rb_allocation_valid = True
    for b in range(num_RBs):
        total_z = np.sum([z_ib_sk_val[i, b, s, k] for s in range(num_slices) for k in range(num_UEs) for i in range(num_RUs)])
        if total_z > 1 + 1e-6:  # Allow small tolerance
            logger.add(f"Constraint violation: RB {b} is allocated to more than one UE (sum = {total_z:.4f})")
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
                    
                    if z_val < 0.5:  # z is 0 (using 0.5 as threshold for binary variables)
                        if mu_val > 1e-6:  # mu should be 0
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
                    else:  # z is 1
                        if abs(mu_val - p_val) > 1e-6:  # mu should equal p
                            logger.add(f"Constraint violation: mu_ib_sk[{i},{b},{s},{k}] = {mu_val:.4f} not equal to p = {p_val:.4f} when z = {z_val:.1f}")
                            mu_constraint_valid = False
    
    logger.add(f"mu = z * p constraint validated: {mu_constraint_valid}")
    
    # 4. Calculate and check data rates (R_sk ≥ R_min * pi_sk)
    rate_constraint_valid = True
    R_sk_val = np.zeros((num_slices, num_UEs))
    
    for s in range(num_slices):
        for k in range(num_UEs):
            R_sk = 0
            for b in range(num_RBs):
                snr = 0
                for i in range(num_RUs):
                    snr += gain[i, b, s, k] * mu_ib_sk_val[i, b, s, k]
                if snr > 0:
                    R_sk += rb_bandwidth * np.log(1 + snr) / np.log(2)
            
            R_sk_val[s, k] = R_sk
            
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
        latency_constraint_valid = validate_latency_constraints(
            num_slices, num_UEs, num_RUs, num_RBs, num_DUs, num_CUs,
            z_ib_sk_val, R_sk_val, phi_j_sk_val, phi_m_sk_val, pi_sk_val,
            c, d_sk, max_latency, L_cu, L_du, rho_du, mu_s, lambda_s, logger
        )
    else:
        logger.add("Latency parameters not provided, skipping latency validation")
    
    logger.add(f"Latency constraint validated: {latency_constraint_valid}")
    
    # 13. Check eMBB data rate upper bound if applicable
    embb_rate_valid = True
    if max_latency is not None:  # If latency constraints are used, check eMBB bounds
        for s in range(num_slices):
            for k in range(num_UEs):
                if pi_sk_val[s, k] > 0.5:  # UE is selected
                    # Assuming slice 1 is eMBB (you may need to adjust this based on your slice definitions)
                    if s == 1:  # eMBB slice
                        embb_max_rate = 0.25  # 0.25ms as mentioned in your code
                        if R_sk_val[s, k] > embb_max_rate + 1e-6:
                            logger.add(f"eMBB rate violation: UE ({s},{k}) rate {R_sk_val[s, k]:.4f} > max {embb_max_rate}")
                            embb_rate_valid = False
    
    logger.add(f"eMBB rate upper bound validated: {embb_rate_valid}")
    
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
                ru_du_valid and du_cu_valid and slice_map_valid and latency_constraint_valid and embb_rate_valid)
    
    logger.add(f"\nAll constraints validated: {all_valid}")
    
    return all_valid, R_sk_val