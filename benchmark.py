def print_results(pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    # In kết quả cho từng biến
    print("Kết quả của pi_sk:")
    for s in range(pi_sk.shape[0]):
        for k in range(pi_sk.shape[1]):
            print(f"pi_sk[{s}, {k}] = {pi_sk[s, k].value}")

    print("\nKết quả của z_ib_sk:")
    for i in range(z_ib_sk.shape[0]):
        for b in range(z_ib_sk.shape[1]):
            for s in range(z_ib_sk.shape[2]):
                for k in range(z_ib_sk.shape[3]):
                    print(f"z_ib_sk[{i}, {b}, {s}, {k}] = {z_ib_sk[i, b, s, k].value}")
                    
    print("\nKết quả của P_ib_sk:")
    for i in range(p_ib_sk.shape[0]):
        for b in range(p_ib_sk.shape[1]):
            for s in range(p_ib_sk.shape[2]):
                for k in range(p_ib_sk.shape[3]):
                    print(f"P_ib_sk[{i}, {b}, {s}, {k}] = {p_ib_sk[i, b, s, k].value}")

    print("\nKết quả của phi_i_sk:")
    for i in range(phi_i_sk.shape[0]):
        for s in range(phi_i_sk.shape[1]):
            for k in range(phi_i_sk.shape[2]):
                print(f"phi_i_sk[{i}, {s}, {k}] = {phi_i_sk[i, s, k].value}")

    """print("\nKết quả của phi_j_sk:")
    for j in range(phi_j_sk.shape[0]):
        for s in range(phi_j_sk.shape[1]):
            for k in range(phi_j_sk.shape[2]):
                print(f"phi_j_sk[{j}, {s}, {k}] = {phi_j_sk[j, s, k].value}")

    print("\nKết quả của phi_m_sk:")
    for m in range(phi_m_sk.shape[0]):
        for s in range(phi_m_sk.shape[1]):
            for k in range(phi_m_sk.shape[2]):
                print(f"phi_m_sk[{m}, {s}, {k}] = {phi_m_sk[m, s, k].value}")

    print("\nKết quả của P_ib_sk:")
    for i in range(p_ib_sk.shape[0]):
        for b in range(p_ib_sk.shape[1]):
            for s in range(p_ib_sk.shape[2]):
                for k in range(p_ib_sk.shape[3]):
                    print(f"P_ib_sk[{i}, {b}, {s}, {k}] = {p_ib_sk[i, b, s, k].value}")

    print("\nKết quả của mu_ib_sk:")
    for i in range(mu_ib_sk.shape[0]):
        for b in range(mu_ib_sk.shape[1]):
            for s in range(mu_ib_sk.shape[2]):
                for k in range(mu_ib_sk.shape[3]):
                    print(f"mu_ib_sk[{i}, {b}, {s}, {k}] = {mu_ib_sk[i, b, s, k].value})"""

def print_results1(pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    print("Kết quả của pi_sk: ", pi_sk)
    print("Kết quả của z_ib_sk: ", z_ib_sk)
    print("Kết quả của phi_i_sk: ", phi_i_sk)
    print("Kết quả của phi_j_sk: ", phi_j_sk)
    print("Kết quả của phi_m_sk: ", phi_m_sk)
    print("Kết quả của P_ib_sk: ", p_ib_sk)
    print("Kết quả của mu_ib_sk: ", mu_ib_sk)

def print_result_short_term(short_pi_sk, short_z_ib_sk, short_p_ib_sk, short_mu_ib_sk):
    try:
        print("Kết quả tối ưu Wireless: ")
        print("Kết quả của short_pi_sk:")
        for s in range(short_pi_sk.shape[0]):
            for k in range(short_pi_sk.shape[1]):
                print(f"short_pi_sk[{s}, {k}] = {short_pi_sk[s, k].value}")

        print("\nKết quả của short_z_ib_sk:")
        for i in range(short_z_ib_sk.shape[0]):
            for b in range(short_z_ib_sk.shape[1]):
                for s in range(short_z_ib_sk.shape[2]):
                    for k in range(short_z_ib_sk.shape[3]):
                        print(f"short_z_ib_sk[{i}, {b}, {s}, {k}] = {short_z_ib_sk[i, b, s, k].value}")
        
        print("\nKết quả của short_p_ib_sk:")
        for i in range(short_p_ib_sk.shape[0]):
            for b in range(short_p_ib_sk.shape[1]):
                for s in range(short_p_ib_sk.shape[2]):
                    for k in range(short_p_ib_sk.shape[3]):
                        print(f"short_p_ib_sk[{i}, {b}, {s}, {k}] = {short_p_ib_sk[i, b, s, k].value}")
        
        print("\nKết quả của short_mu_ib_sk:")
        for i in range(short_mu_ib_sk.shape[0]):
            for b in range(short_mu_ib_sk.shape[1]):
                for s in range(short_mu_ib_sk.shape[2]):
                    for k in range(short_mu_ib_sk.shape[3]):
                        print(f"short_mu_ib_sk[{i}, {b}, {s}, {k}] = {short_mu_ib_sk[i, b, s, k].value}")

    except Exception as e:
        print(f"Lỗi khi đang in kết quả: {e}")