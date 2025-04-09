#noise_power_watts = 1e-10
#pp = max_tx_power_watts/((num_RBs))
#obj = obj = 40000*(11.00*x_em[frame_num] + 12.00*x_ur[frame_num] + 60.00*x_mm[frame_num]) - (temp10 + temp11) + 1000*temp12
#U_em/U_ur/U_mm:8/10/20
#x= [0,-650,-650,650,650]
#y= [0,650,-650,650,-650]
#r_outdoor= 1000              #Outside radious for RU (meter)
#r_indoor= 100 #Inside radious for RU (meter)

import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy.linalg import norm
import cvxpy as cp
import datetime
#from pyomo.solvers.plugins.solver.BONMIN import BONMIN

# Simulation settings

num_RBs = 100  
rb_bandwidth = 180e3 
rho = 0.99                              #correlation factor for channel
L_ur = 0.25                             #delay requirenment of urllc (ms)
R_em = 1e6                              #minimum requirenment throughput for embb (10Mbps)
gamma_mm = 3.16
gamma_mm_sca = 7
# Maximum transmission power
max_tx_power_dbm = 43 # dBm
max_tx_power_watts = 10**((max_tx_power_dbm)/10) # in mWatts  
noise_power_watts = 1e-10 # in mWatts

tti_size = 7
error_prob = 1e-3

num_antennas = 8  

# Propagation model params
path_loss_exp = 37.6
path_loss_ref = 128.1

shadow_fading_std = 8 

# Packet sizes 
urllc_packet_size = 32  
embb_packet_size = 50e3
mtc_packet_size = 16  

# Traffic model
urllc_traffic = 'Poisson'  

num_RUs = 5
num_DUs = 3

num_slices_per_service = 1
U_em = 4*num_slices_per_service  #4 6 8
U_ur = 6*num_slices_per_service  #6 8 10
U_mm = 12*num_slices_per_service  #8 10 12
U = U_em + U_ur + U_mm

slot_dur = 1e-3
frame_dur = 10e-3
num_slots_per_frame = 10
num_frames = 1000

em_dims = num_RUs*U_em*num_RBs*num_slots_per_frame
ur_dims = num_RUs*U_ur*num_RBs*num_slots_per_frame
mm_dims = num_RUs*U_mm*num_RBs*num_slots_per_frame

alpha = 40000
beta = 10

u_du = 10000
u_cu = 50000
phi = 0.5
eta = 0.5
dummy = 100

def generate_position1():
    np.random.seed(1)
    x= [0,-650,-650,650,650]
    y= [0,650,-650,650,-650]
    r_outdoor= 1000              #Outside radious for RU (meter)
    r_indoor= 100   
    n= 1                        #number of sectors

    teta= np.arange(0, 2*math.pi, 0.01)
    x1= r_outdoor* np.cos(teta)+ x[0]
    y1= r_outdoor* np.sin(teta)+ y[0]
    x2= r_indoor* np.cos(teta)+ x[0]
    y2= r_indoor* np.sin(teta)+ x[0]
    plt.rcParams['font.sans-serif'] = "Cambria"
    plt.rcParams['font.family'] = "sans-serif"
    plt.figure(figsize=(8, 9))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(np.arange(-1000, 1001, 500),fontsize = 15)
    plt.yticks(np.arange(-1000, 1001, 500),fontsize = 15)
    plt.scatter(x[0], y[0], marker='^', s=100, color= 'green')
    plt.text(x[0], y[0], 'RU', ha='right', va='bottom',fontsize=10)
    plt.plot(x1, y1) 
    plt.xlim(-1000, 1001)
    plt.ylim(-1000, 1001)
    plt.xlabel("x-coordinate(m)",fontsize = 25)
    plt.ylabel("y-coordinate(m)",fontsize = 25)
    plt.plot(x2, y2, color= 'gray') 
    tet= np.linspace(0,2*math.pi,n+1)
    xi= r_outdoor* np.cos(tet)+x[0]
    yi= r_outdoor* np.sin(tet)+y[0]

    #-------------------------divide the circle to n sectors----------------------
    for i in range(len(xi)):
        plt.plot([x[0], xi[i]], [y[0], yi[i]], color= 'gray', linestyle = '--', linewidth=0.9)

    plt.scatter(x, y, marker='^', s=100, color= 'green', label ="RUs")
    #--------------------------randomly user distribution-------------------------
    n_em, n_ur, n_mm = int(U_em/n), int(U_ur/n), int(U_mm/n)
    D_em, D_ur, D_mm = np.zeros((num_RUs,U_em)), np.zeros((num_RUs,U_ur)), np.zeros((num_RUs,U_mm))

    for i in range(n):
        for attempt in range(20000):
            d_em = (r_outdoor-r_indoor)* np.random.random(n_em)+ r_indoor
            d_ur = (r_outdoor-r_indoor)* np.random.random(n_ur)+ r_indoor
            d_mm = (r_outdoor-r_indoor)* np.random.random(n_mm)+ r_indoor
            t_em = (tet[i+1] - tet[i]) * np.random.random(n_em) + tet[i]
            t_ur = (tet[i+1] - tet[i]) * np.random.random(n_ur) + tet[i]
            t_mm = (tet[i+1] - tet[i]) * np.random.random(n_mm) + tet[i]
            x_em = x[0] + d_em* np.cos(t_em)  #the x coordinate of int(U_em/n) user at sector i
            y_em = y[0] + d_em* np.sin(t_em)  #the y coordinate of int(U_em/n) user at sector i
            x_ur = x[0] + d_ur* np.cos(t_ur)
            y_ur = y[0] + d_ur* np.sin(t_ur)
            x_mm = x[0] + d_mm* np.cos(t_mm)
            y_mm = y[0] + d_mm* np.sin(t_mm)
            temp_em = 0
            temp_mm = 0
            temp_ur = 0
            for m in range(num_RUs):
                D_em[m,i*n_em:(i*n_em)+n_em] = np.sqrt((x_em-x[m])**2+(y_em-y[m])**2)
                D_ur[m,i*n_ur:(i*n_ur)+n_ur] = np.sqrt((x_ur-x[m])**2+(y_ur-y[m])**2) 
                D_mm[m,i*n_mm:(i*n_mm)+n_mm] = np.sqrt((x_mm-x[m])**2+(y_mm-y[m])**2)
            dis_ur_m = np.zeros((U_ur, 3), dtype=int)
            dis_mm_m = np.zeros((U_mm, 3), dtype=int)
            dis_em_m = np.zeros((U_em, 3), dtype=int)
            for k in range(U_mm):
                dis_mm_m[k] = np.argsort(D_mm[:, k])[:3]
            for k in range(U_em):
                dis_em_m[k] = np.argsort(D_em[:, k])[:3]
            for k in range(U_ur):
                dis_ur_m[k] = np.argsort(D_ur[:, k])[:3]
            for k in range(U_em):
                if D_em[dis_em_m[k,2], k] > 650:
                    temp_em += 1
            for k in range(U_ur):
                if D_ur[dis_ur_m[k,2], k] > 650:
                    temp_ur += 1
            for k in range(U_mm):
                if D_mm[dis_mm_m[k,2], k] > 650:
                    temp_mm += 1
            
            if (D_em >= r_indoor).all() and (D_mm >= r_indoor).all() and (D_ur >= r_indoor).all() and temp_em==U_em and temp_ur==U_ur and temp_mm == U_mm:  # Check if all distances are greater than r_indoor
                break  # Valid positions found
        else:
            raise RuntimeError("Unable to find valid positions after 100 attempts.")
        plt.scatter(x_em, y_em, color= 'blue', marker="o", label ="EMBB users")
        plt.scatter(x_ur, y_ur, color= 'orange', marker="*", label ="URLLC users")
        plt.scatter(x_mm, y_mm, color= 'red', marker="s", label ="MMTC users")
        
        for m in range(num_RUs):
            D_em[m,i*n_em:(i*n_em)+n_em] = np.sqrt((x_em-x[m])**2+(y_em-y[m])**2)
            D_ur[m,i*n_ur:(i*n_ur)+n_ur] = np.sqrt((x_ur-x[m])**2+(y_ur-y[m])**2) 
            D_mm[m,i*n_mm:(i*n_mm)+n_mm] = np.sqrt((x_mm-x[m])**2+(y_mm-y[m])**2)
    #plt.text(x_ur, y_ur, 'uRLLC', ha='right', va='bottom',fontsize=10)
    #plt.savefig('topology.pdf', dpi=300, bbox_inches='tight')
    #np.savetxt("sca/para/Distance_ur.csv", D_ur, delimiter=',')
    #np.savetxt("sca/para/Distance_em.csv", D_em, delimiter=',')
    #np.savetxt("sca/para/Distance_mm.csv", D_mm, delimiter=',')
    current_datetime = datetime.datetime.now()
    datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
    fig_name = fr"network.pdf"
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol = 2, fontsize = 20)
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(fig_name)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    return D_ur, D_em, D_mm

def channel_gain1(dis_em, dis_ur, dis_mm):
    g_em = np.zeros((num_RUs,U_em,num_RBs, num_slots_per_frame), dtype=np.float64)
    g_ur = np.zeros((num_RUs,U_ur,num_RBs, num_slots_per_frame), dtype=np.float64)
    g_mm = np.zeros((num_RUs,U_mm,num_RBs, num_slots_per_frame), dtype=np.float64)
    

    path_loss_em = 128.1 + 37.6*np.log10(dis_em*1e-3)
    path_loss_mm = 128.1 + 37.6*np.log10(dis_mm*1e-3)
    path_loss_ur = 128.1 + 37.6*np.log10(dis_ur*1e-3)

    # Simulate h as complex Gaussian with zero mean and unit variance
   
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_em):
                for t in range(num_slots_per_frame):
            
                    channel_gain = 10**(-path_loss_em[i, k]/(10))/noise_power_watts
                    h = np.sqrt(channel_gain)*np.sqrt(1/2)*(np.random.rand(num_antennas)+ 1j*np.random.rand(num_antennas))
                    g_em[i,k,b,t]= (norm(h,2)**2)

    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_ur):
                for t in range(num_slots_per_frame):
                    channel_gain = 10**(-path_loss_ur[i, k]/(10))/noise_power_watts
                    h = np.sqrt(channel_gain)*np.sqrt(1/2)*(np.random.rand(num_antennas)+ 1j*np.random.rand(num_antennas))
                    g_ur[i,k,b,t]= (norm(h,2)**2)

    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_mm):
                for t in range(num_slots_per_frame):
                    channel_gain = 10**(-path_loss_mm[i, k]/(10))/noise_power_watts
                    h = np.sqrt(channel_gain)*np.sqrt(1/2)*(np.random.rand(num_antennas)+ 1j*np.random.rand(num_antennas))
                    g_mm[i,k,b,t]= (norm(h,2)**2)
    
    np.savetxt("sca/para/G_ur.csv", g_ur[0][0], delimiter=',')
    np.savetxt("sca/para/G_em.csv", g_em[0][0], delimiter=',')
    np.savetxt("sca/para/G_mm.csv", g_mm[0][0], delimiter=',')
    return g_em, g_ur, g_mm

def generate_correlated_poisson(length, rate, correlation_factor):
    uncorrelated = np.random.poisson(rate, length) # Generate uncorrelated Poisson process
    correlated = np.zeros(length)                  # Create a correlated time series
    correlated[0] = uncorrelated[0]
    # Generate correlated values
    for i in range(1, length):
        correlated[i] = np.sqrt(correlation_factor)*correlated[i-1] + np.sqrt(1-correlation_factor)*uncorrelated[i]
    return correlated

def generate_lambda():
    mean_rate_ur = 0.3
    mean_rate_mm = 0.3
    mean_rate_em = 5
    correlation_factor = 0.98   
    data = np.zeros((num_frames, U))
    for u in range(U_ur):
        data[:,u] = generate_correlated_poisson(num_frames, mean_rate_ur, correlation_factor)
    for u in range(U_em):
        data[:,U_ur+u] = generate_correlated_poisson(num_frames, mean_rate_em, correlation_factor)
    for u in range(U_mm):
        data[:,U_ur+U_em+u] = generate_correlated_poisson(num_frames, mean_rate_mm, correlation_factor)
    data_ur = data[:,0:U_ur]
    data_em = data[:,U_ur:U_ur+U_em]
    data_mm = data[:,U_ur+U_em::]
    #save data
    #np.savetxt("sca/para/data_ur.csv", data_ur, delimiter=',')
    #np.savetxt("sca/para/data_em.csv", data_em, delimiter=',')
    #np.savetxt("sca/para/data_mm.csv", data_mm, delimiter=',')
    return data_ur, data_em, data_mm

def short_subproblem(dis_ur, y ,z_em, z_ur, z_mm, g_em, g_ur, g_mm, data_ur, frame_num, z2_ur, dis_mm_m, dis_ur_m, t):
    try:
        #p_em, p_ur, p_mm = [], [],[]
        p_em = np.empty((num_RUs, U_em, num_RBs), dtype=object)
        p_ur = np.empty((num_RUs, U_ur, num_RBs), dtype=object)
        p_mm = np.empty((num_RUs, U_mm, num_RBs), dtype=object)
        for i in range(num_RUs):
            for b in range(num_RBs):
                #for t in range(num_slots_per_frame):
                for k in range(U_em):
                    p_em[i,k,b]= cp.Variable()
                for k in range(U_ur):
                    p_ur[i,k,b]= cp.Variable()
                for k in range(U_mm):
                    p_mm[i,k,b]= cp.Variable()
        obj = cp.Variable()
        objective= cp.Maximize(obj)
        constraint= []
        constraint.append( obj >= 0)

        for i in range(num_RUs):
            for b in range(num_RBs):  
                for k in range(U_em):  
                    constraint.append(p_em[i,k,b] >=0)
                for k in range(U_ur):  
                    constraint.append(p_ur[i,k,b] >=0)
                for k in range(U_mm):  
                    constraint.append(p_mm[i,k,b] >=0)
        
        
        for i in range(num_RUs):
            temp0 = 0
            for b in range(num_RBs):
                for k in range(U_em):  
                    temp0 += p_em[i,k,b]*z_em[i,k,b]
                for k in range(U_ur):  
                    temp0 += p_ur[i,k,b]*z_ur[i,k,b]
                    #temp0 += p_ur[i,k,b,t]
                for k in range(U_mm):  
                    temp0 += p_mm[i,k,b]*z_mm[i,k,b]
                    #temp0 += p_mm[i,k,b,t]
            constraint.append( temp0 <= max_tx_power_watts)
        
        for k in range(U_em):
            temp1 = 0
            for b in range(num_RBs):
                temp2 = 0
                for i in range(num_RUs): 
                    temp2 += p_em[i,k,b]*z_em[i,k,b]*g_em[i,k,b,t]
                temp1 += rb_bandwidth*(cp.log(1+ temp2)/0.3)
            constraint.append( temp1 - R_em >= obj)
        
        for k in range(U_mm):
            temp5 = 0
            for i in dis_mm_m[k]:
                for b in range(num_RBs):
                    temp5 += p_mm[i,k,b]*z_mm[i,k,b]*g_mm[i,k,b,t]
            constraint.append( temp5 - gamma_mm >= obj)
        for k in range(U_ur):
            temp5a = 0
            for i in dis_ur_m[k]:
                for b in range(num_RBs):
                    temp5a += p_ur[i,k,b]*z_ur[i,k,b]*g_ur[i,k,b,t]
            constraint.append( temp5a - gamma_mm >= obj)
        
        data_ur1 = 0
        for k in range(U_ur):
            data_ur1 += data_ur[frame_num,k]
        L_cu = 1/(u_cu - data_ur1)
        L_du = 1/(u_du - data_ur1)
        rho_du = data_ur1/u_du
        A = np.zeros((num_DUs,num_RUs,U_ur,num_RBs))
        A_m = np.zeros((num_DUs,num_RUs,U_ur))
        for j in range (num_DUs):
            for i in range(num_RUs):
                for k in range(U_ur):
                    for b in range(num_RBs):
                        A[j,i,k,b] = urllc_packet_size/(L_ur - z_ur[i,k,b]*(dis_ur[i,k]/(3*1e8)+ rho_du/(u_du - data_ur1)) - L_cu - y[j]*L_du)
                    A_m[j,i,k] = np.max(A[j,i,k,:])

        
        for j in range(num_DUs):
            for k in range(U_ur):
                temp3 = 0
                for b in range(num_RBs):
                    temp4 = 0
                    for i in range(num_RUs):
                        temp4 += p_ur[i,k,b]*z_ur[i,k,b]*g_ur[i,k,b,t]
                    if z_ur[i,k,b] > 0:
                        temp3 += rb_bandwidth*(cp.log(1+ temp4)/0.3 - z2_ur[k,b]*(stats.norm.ppf(1 - error_prob))/cp.sqrt(urllc_packet_size))
                for i in range(num_RUs):
                    constraint.append( temp3 - A_m[j,i,k] >= obj)
        
        # Solve problem
        prob = cp.Problem(objective, constraint)
        prob.solve(solver = 'MOSEK',verbose = True) 
        output = prob.status

    except cp.SolverError:
        output = 'non_feasible' 
    print('Output is:', output)
    p_em1 = np.zeros((num_RUs, U_em, num_RBs))
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_em):
                p_em1[i,k,b] = p_em[i,k,b].value
    p_ur1 = np.zeros((num_RUs, U_ur, num_RBs))
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_ur):
                p_ur1[i,k,b] = p_ur[i,k,b].value
    p_mm1 = np.zeros((num_RUs, U_mm, num_RBs))
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_mm):
                p_mm1[i,k,b] = p_mm[i,k,b].value
    p_ru = []
    for i in range(num_RUs):
        temp = 0
        for b in range(num_RBs):
            for k in range(U_em):
                temp += p_em1[i,k,b]*z_em[i,k,b]
            for k in range(U_ur):
                temp += p_ur1[i,k,b]*z_ur[i,k,b]
            for k in range(U_mm):
                temp += p_mm1[i,k,b]*z_mm[i,k,b]
        p_ru.append(temp)

    R_user_em = []
    for k in range(U_em):
        temp3 = 0
        for b in range(num_RBs):
            temp4 = 0
            for i in range(num_RUs):
                temp4 += p_em1[i,k,b]*z_em[i,k,b]*g_em[i,k,b,t]
            temp3 += rb_bandwidth*(np.log(1+ temp4)/0.3)
        R_user_em.append(temp3)
        print(f"R user {k} = {temp3}")
        #print(z1_ur[k,0]*(stats.norm.ppf(1 - error_prob))/np.sqrt(urllc_packet_size))
    
    R_user_ur = []
    for k in range(U_ur):
        temp5 = 0
        for b in range(num_RBs):
            temp6 = 0
            for i in range(num_RUs):
                temp6 += p_ur1[i,k,b]*z_ur[i,k,b]*g_ur[i,k,b,t]
            temp5 += rb_bandwidth*(np.log(1+ temp6)/0.3 - z2_ur[k,b]*(stats.norm.ppf(1 - error_prob))/np.sqrt(urllc_packet_size))
        R_user_ur.append(temp5)

    L_user_ur = []
    L = np.zeros((num_DUs,num_RUs,U_ur,num_RBs))
    for k in range(U_ur):
        for j in range (num_DUs):
            for i in range(num_RUs):
                for b in range(num_RBs):
                    L[j,i,k,b] = z_ur[i,k,b]*(dis_ur[i,k]/(3*1e8)+ rho_du/(u_du - data_ur1)) + L_cu + y[j]*L_du + urllc_packet_size/R_user_ur[k]
        L_user_ur.append(np.max(L[:,:,k,:]))
    temp = 0
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_em):
                temp += p_em1[i,k,b]*z_em[i,k,b]
            for k in range(U_mm):
                temp += p_mm1[i,k,b]*z_mm[i,k,b] 
            for k in range(U_ur):
                temp += p_ur1[i,k,b]*z_ur[i,k,b]
    mean_p = temp/U

    '''
    for k in range(U_em):
        #for i in range(num_RUs):
        filename = fr"sca/power/p_em_{k}.csv"
        np.savetxt(filename, p_em1[:,k,:], delimiter=',')
    for k in range(U_ur):
        #for i in range(num_RUs):
        filename = fr"sca/power/p_ur_{k}.csv"
        np.savetxt(filename, p_ur1[:,k,:], delimiter=',')
    for k in range(U_mm):
        #for i in range(num_RUs):
        filename = fr"sca/power/p_mm_{k}.csv"
        np.savetxt(filename, p_mm1[:,k,:], delimiter=',')
    '''
    return p_em1, p_ur1, p_mm1, R_user_em, L_user_ur, mean_p, p_ru

def long_sub_SCA(p_em, g_em, p_ur, g_ur, p_mm, g_mm, data_ur, dis_ur, frame_num, y1, z_em0, z_ur0, z_mm0, z1_ur0,z2_ur0, y0, x_em0,x_ur0,x_mm0, u2_0,dis_em_m,dis_ur_m,dis_mm_m):
    try:
        z_em, z_ur, z_mm = {}, {}, {}
        for i in range(num_RUs):
           for b in range(num_RBs):
                for k in range(U_em):
                    z_em[(i,k,b)]= cp.Variable()
                for k in range(U_ur):
                    z_ur[(i,k,b)]= cp.Variable()
                for k in range(U_mm):
                    z_mm[(i,k,b)]= cp.Variable()
        
        z1_ur = {}
        for k in range(U_ur):
            for i in range(num_RUs):
                z1_ur[(i,k)]= cp.Variable()
        z2_ur = {}
        for k in range(U_ur):
            for b in range(num_RBs):
                z2_ur[(k,b)]= cp.Variable()
        
        y = {}
        for j in range(num_DUs):
            y[j] =  cp.Variable()
        
        x_em, x_ur, x_mm = {}, {}, {}
        x_em =  cp.Variable()
        x_ur =  cp.Variable()
        x_mm =  cp.Variable()

        c_j = {}
        for j in range(num_DUs):
            c_j[j] = cp.Variable()
        
        u2 = {}
        u2 = cp.Variable()

        u1 = {}
        u1 =  cp.Variable()

        constraint= []
        
        #constraint 23g,h
        for i in range(num_RUs):
            for b in range(num_RBs): 
                for k in range(U_em):
                    constraint.append(z_em[(i,k,b)] >= 0)
                    constraint.append(z_em[(i,k,b)] <= 1)
                for k in range(U_ur):
                    constraint.append(z_ur[(i,k,b)] >= 0)
                    constraint.append(z_ur[(i,k,b)] <= 1)
                for k in range(U_mm):
                    constraint.append(z_mm[(i,k,b)] >= 0)
                    constraint.append(z_mm[(i,k,b)] <= 1)
        
        
        for j in range(num_DUs):
            constraint.append(y[j] >= 0)
            constraint.append(y[j] <= 1)
        for k in range(U_ur):
            for i in range(num_RUs):
                constraint.append(z1_ur[(i,k)] >= 0)
                constraint.append(z1_ur[(i,k)] <= 1)
        
        for k in range(U_ur):
            for b in range(num_RBs):
                constraint.append(z2_ur[(k,b)] >= 0)
                constraint.append(z2_ur[(k,b)] <= 1)
        
        constraint.append(x_em >= 0)
        constraint.append(x_em <= 1)
        constraint.append(x_ur >= 0)
        constraint.append(x_ur <= 1)
        constraint.append(x_mm >= 0)
        constraint.append(x_mm <= 1)
        constraint.append(u1 >= 1e-4)
        constraint.append(u2 >= 0)
        constraint.append(u2 <= 1)

            
        temp1 = 0
        for j in range(num_DUs):
            temp1 += y[j]*0.05 
        constraint.append(temp1 <= 1)
        temp1b = 0
        for j in range(num_DUs):
            temp1b += y[j]
        constraint.append(temp1b >= 1)
        #constraint 23b
        #constraint.append(x_em[frame_num]*0.1 + x_ur[frame_num]*0.1 + x_mm[frame_num]*0.1 <= 1)

        #constraint.append(x_em[frame_num] + x_ur[frame_num] + x_mm[frame_num] >= 2.5)
        for i in range(num_RUs):
            for b in range(num_RBs):
                temp0 = 0
                for k in range(U_em):
                    temp0 += z_em[(i,k,b)]
                for k in range(U_ur):
                    temp0 += z_ur[(i,k,b)]
                for k in range(U_mm):
                    temp0 += z_mm[(i,k,b)] 
                constraint.append(temp0 <= 1)
        '''
        for k in range(U_ur):
            temp0a = 0
            for i in range(num_RUs):
                for b in range(num_RBs):
                    temp0a += z_ur[(i,k,b)]
            constraint.append(temp0a <= 40)
        
        for k in range(U_em):
            temp0b = 0
            for i in range(num_RUs):
                for b in range(num_RBs):
                    temp0b += z_em[(i,k,b)]
            constraint.append(temp0b <= 35)
        for k in range(U_mm):
            temp0c = 0
            for i in range(num_RUs):
                for b in range(num_RBs):
                    temp0c += z_mm[(i,k,b)]
            constraint.append(temp0c <= 35)
        '''
        #constraint 23d
        for b in range(num_RBs): 
            for k in range(U_em):
                for i in range(num_RUs):
                    for j in range (i,num_RUs):
                        constraint.append(z_em[(i,k,b)] == z_em[(j,k,b)])
            for k in range(U_ur):
                for i in range(num_RUs):
                    for j in range (i,num_RUs):
                        constraint.append(z_ur[(i,k,b)] == z_ur[(j,k,b)])
            for k in range(U_mm):
                for i in range(num_RUs):
                    for j in range (i,num_RUs):
                        constraint.append(z_mm[(i,k,b)] == z_mm[(j,k,b)])

        
        for k in range(U_ur):
            for i in range(num_RUs):
                temp1a = 0
                for b in range(num_RBs): 
                    temp1a += z_ur[(i,k,b)]
                constraint.append(temp1a/num_RBs <= z1_ur[(i,k)])

        
        for k in range(U_ur):
            for i in range(num_RUs):
                temp2a = 0
                for b in range(num_RBs): 
                    temp2a += z_ur[(i,k,b)]
                constraint.append(z1_ur[(i,k)] - temp2a/(num_RBs) <= 0.99999)
        
        #constraint 26a
        for k in range(U_ur):
            for b in range(num_RBs):
                temp3a = 0
                for i in range(num_RUs):
                    temp3a += z_ur[(i,k,b)]
                constraint.append(z2_ur[(k,b)] == temp3a/num_RUs)
        for k in range(U_ur):
            temp3 = 0
            for b in range(num_RBs):
                temp4 = 0
                for i in range(num_RUs): 
                    temp4 += p_ur[i,k,b]*z_ur[(i,k,b)]*g_ur[i,k,b,num_slots_per_frame-1]
                temp3 += rb_bandwidth*(cp.log(1+ temp4)/0.3 - z2_ur[(k,b)]*(stats.norm.ppf(1 - error_prob))/cp.sqrt(urllc_packet_size))
            constraint.append( temp3 >= cp.inv_pos(u1))
        
        data_ur1 = 0
        for k in range(U_ur):
            data_ur1 += data_ur[frame_num,k]
        L_cu = 1/(u_cu - data_ur1)
        L_du = 1/(u_du - data_ur1)
        rho_du = data_ur1/u_du

        #constraint 26b
        for i in range(num_RUs):
            for j in range(num_DUs):
                for k in range(U_ur): 
                    #temp5 = z_ur[(i,k,b)]*(dis_ur[i,k]/(3*1e8)+ rho_du/(u_du - data_ur1)) + urllc_packet_size*u1 + L_cu + y[j]*L_du
                    temp5 = z1_ur[(i,k)]*(dis_ur[i,k]/(3*1e8)+ rho_du/(u_du - data_ur1)) + urllc_packet_size*u1 + L_cu + y[j]*L_du
                    constraint.append( temp5 <= L_ur*x_ur)
                    #constraint.append( temp5 <= L_ur)
        
        #constraint 28b
        for k in range(U_em):
            temp6 = 0
            for b in range(num_RBs):
                temp7 = 0
                for i in range(num_RUs): 
                    temp7 += p_em[i,k,b]*z_em[(i,k,b)]*g_em[i,k,b,num_slots_per_frame-1]
                temp6 += rb_bandwidth*(cp.log(1+ temp7)/0.3)
            constraint.append(temp6 >= R_em*x_em)
            #constraint.append(temp6 >= R_em)
        
        
        for k in range(U_mm):
            for i in dis_mm_m[k]:
                temp9 = 0
                for b in range(num_RBs):
                    temp9 += p_mm[i,k,b]*z_mm[(i,k,b)]*g_mm[i,k,b,num_slots_per_frame-1]
                constraint.append( temp9 >= gamma_mm_sca*x_mm)
        
        for k in range(U_em):
            for i in dis_em_m[k]:
                temp9a = 0
                for b in range(num_RBs):
                    temp9a += p_em[i,k,b]*z_em[(i,k,b)]*g_em[i,k,b,num_slots_per_frame-1]
                constraint.append( temp9a >= gamma_mm_sca*x_em)
        

        for k in range(U_ur):
            for i in dis_ur_m[k]:
                temp9b = 0
                for b in range(num_RBs):
                    temp9b += p_ur[i,k,b]*z_ur[(i,k,b)]*g_ur[i,k,b,num_slots_per_frame-1]
                constraint.append( temp9b >= gamma_mm_sca*x_ur)
            #constraint.append( temp9 >= gamma_mm)
        #constraint 27a-d
        for j in range(num_DUs):
            constraint.append(c_j[j] >= 0)
            if frame_num ==0:
                constraint.append(c_j[j] >= y[j])
                constraint.append(c_j[j] <= y[j] + 10*(1 - u2))
            else:
                constraint.append(c_j[j] >= y[j] - y1[j])
                constraint.append(c_j[j] <= y[j] - y1[j] + 10*(1 - u2))
            constraint.append(c_j[j] <= 10*u2)
        
        temp10 = 0
        for j in range(num_DUs):
            temp10 += c_j[j]
        
        temp11 = 0
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(U_em):
                    temp11 += z_em[(i,k,b)]
                for k in range(U_ur):
                    temp11 += z_ur[(i,k,b)]
                for k in range(U_mm):
                    temp11 += z_mm[(i,k,b)]
        temp12 = 0
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(U_em): 
                    temp12 += z_em[(i,k,b)]*(2*z_em0[i,k,b]-1) - z_em0[i,k,b]**2
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(U_ur): 
                    temp12 += z_ur[(i,k,b)]*(2*z_ur0[i,k,b]-1) - z_ur0[i,k,b]**2
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(U_mm): 
                    temp12 += z_mm[(i,k,b)]*(2*z_mm0[i,k,b]-1) - z_mm0[i,k,b]**2
        for k in range(U_ur):
            for i in range(num_RUs):
                temp12 += z1_ur[(i,k)]*(2*z1_ur0[i,k]-1) - z1_ur0[i,k]**2
        for k in range(U_ur):
            for b in range(num_RBs):
                temp12 += z2_ur[(k,b)]*(2*z2_ur0[k,b]-1) - z2_ur0[k,b]**2
        for j in range(num_DUs):
            temp12 += y[j]*(2*y0[j]-1) - y0[j]**2
        temp12 += u2*(2*u2_0 - 1) - u2_0**2
        temp12 += x_em*(2*x_em0 - 1) - x_em0**2
        temp12 += x_ur*(2*x_ur0 - 1) - x_ur0**2
        temp12 += x_mm*(2*x_mm0 - 1) - x_mm0**2
        
        obj = alpha*(11.00*x_em + 12.00*x_ur + 60.00*x_mm) - beta*(temp10 + temp11) + 1000*cp.sum(temp12)#x_em 9.40 x_mm 43.5
        #obj = alpha*(15.00*x_em + 40.00*x_ur + 50.00*x_mm) - beta*(temp10 + temp11) + 1000*cp.sum(temp12)#x_em 9.40 x_mm 43.5
        #obj = 10000*(30*x_em + 20*x_ur + 40*x_mm) - (temp10 + temp11) + 1000*cp.sum(temp12)
        objective= cp.Maximize(obj)

        # Solve problem
        prob = cp.Problem(objective, constraint)
        prob.solve(solver = 'MOSEK',verbose = True) 
        output = prob.status

    except cp.SolverError:
        output = 'non_feasible'
    print('Output is:', output)
    z_em1 = np.zeros((num_RUs,U_em,num_RBs))
    
    for b in range(num_RBs):
        for k in range(U_em):
            for i in range(num_RUs):
                z_em1[i,k,b] = z_em[(i,k,b)].value
    z_ur1 = np.zeros((num_RUs,U_ur,num_RBs))
    
    for b in range(num_RBs):
        for k in range(U_ur):
            for i in range(num_RUs):
                z_ur1[i,k,b] = z_ur[(i,k,b)].value
    z_mm1 = np.zeros((num_RUs,U_mm,num_RBs))
    
    for b in range(num_RBs):
        for k in range(U_mm):
            for i in range(num_RUs):
                z_mm1[i,k,b] = z_mm[(i,k,b)].value
    z1_ur1 = np.zeros((num_RUs,U_ur))
    for k in range(U_ur):
        for i in range(num_RUs):
            z1_ur1[i,k] = z1_ur[(i,k)].value
    z2_ur1 = np.zeros((U_ur,num_RBs))
    for k in range(U_ur):
        for b in range(num_RBs):
            z2_ur1[k,b] = z2_ur[(k,b)].value
    y1 = np.zeros((num_DUs))
    for j in range(num_DUs):
        y1[j] = y[j].value
    u2_1 = u2.value
    x_em1 = x_em.value
    x_ur1 = x_ur.value
    x_mm1 = x_mm.value
    optimal_value = prob.value
    print("x_em = ", x_em.value)
    print("x_ur = ", x_ur.value)
    print("x_mm = ", x_mm.value)
    print("Object value:",prob.value)


    z_em0 = np.zeros((num_RUs,U_em,num_RBs))
    for b in range(num_RBs):
        for k in range(U_em):
            for i in range(num_RUs):
                z_em0[i,k,b] = math.floor(z_em[(i,k,b)].value+0.5)

    z_ur0 = np.zeros((num_RUs,U_ur,num_RBs))
    for b in range(num_RBs):
        for k in range(U_ur):
            for i in range(num_RUs):
                z_ur0[i,k,b] = math.floor(z_ur[(i,k,b)].value+0.5)
    
    z_mm0 = np.zeros((num_RUs,U_mm,num_RBs))
    for b in range(num_RBs):
        for k in range(U_mm):
            for i in range(num_RUs):
                z_mm0[i,k,b] = math.floor(z_mm[(i,k,b)].value+0.5)
    
    temp10 = 0
    for j in range(num_DUs):
        temp10 += c_j[j].value
        
    temp11 = 0
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_em):
                temp11 += z_em0[i,k,b]
            for k in range(U_ur):
                temp11 += z_ur0[i,k,b]
            for k in range(U_mm):
                temp11 += z_mm0[i,k,b]
    
    op_value = alpha*(11.00*math.floor(x_em.value+0.5) + 12.00*math.floor(x_ur.value+0.5) + 60.00*math.floor(x_mm.value+0.5)) - beta*(temp10 + temp11)
    print("Object value 1:",op_value)
    return z_em1, z_ur1, z_mm1, z1_ur1, z2_ur1,y1, u2_1, x_em1,x_ur1,x_mm1, optimal_value, op_value

def long_sub_SCA_0(z_em, z_ur, z_mm, p_em, g_em, p_ur, g_ur, p_mm, g_mm, data_ur, dis_ur, frame_num,y1):
    try:
        z1_ur = {}
        for k in range(U_ur):
            for i in range(num_RUs):
                z1_ur[(i,k)]= cp.Variable()
        z2_ur = {}
        for k in range(U_ur):
            for b in range(num_RBs):
                z2_ur[(k,b)]= cp.Variable()
        
        y = {}
        for j in range(num_DUs):
            y[j] =  cp.Variable()

        x_em, x_ur, x_mm = {}, {}, {}
        x_em =  cp.Variable()
        x_ur =  cp.Variable()
        x_mm =  cp.Variable()

        c_j = {}
        for j in range(num_DUs):
            c_j[j] = cp.Variable()
        
        u2 = {}
        u2 = cp.Variable()

        u1 = {}
        u1 =  cp.Variable()

        constraint= []
        
        for k in range(U_ur):
            for i in range(num_RUs):
                constraint.append(z1_ur[(i,k)] >= 0)
                constraint.append(z1_ur[(i,k)] <= 1)
        
        for k in range(U_ur):
            for b in range(num_RBs):
                constraint.append(z2_ur[(k,b)] >= 0)
                constraint.append(z2_ur[(k,b)] <= 1)


        for j in range(num_DUs):
            constraint.append(y[j] >= 0)
            constraint.append(y[j] <= 1)
        
        constraint.append(x_em >= 0)
        constraint.append(x_em <= 1)
        constraint.append(x_ur >= 0)
        constraint.append(x_ur <= 1)
        constraint.append(x_mm >= 0)
        constraint.append(x_mm <= 1)

        constraint.append(u1 >= 1e-4)
        constraint.append(u2 >= 0)
        constraint.append(u2 <= 1)

            
        temp1 = 0
        for j in range(num_DUs):
            temp1 += y[j]*0.05 
        constraint.append(temp1 <= 1)

        
        for k in range(U_ur):
            for i in range(num_RUs):
                temp1a = 0
                for b in range(num_RBs): 
                    temp1a += z_ur[(i,k,b)]
                constraint.append(temp1a/num_RBs <= z1_ur[(i,k)])

        
        for k in range(U_ur):
            for i in range(num_RUs):
                temp2a = 0
                for b in range(num_RBs): 
                    temp2a += z_ur[(i,k,b)]
                constraint.append(z1_ur[(i,k)] - temp2a/(num_RBs) <= 0.99999)
        
        for k in range(U_ur):
            for b in range(num_RBs):
                temp3a = 0
                for i in range(num_RUs):
                    temp3a += z_ur[(i,k,b)]
                constraint.append(z2_ur[(k,b)] == temp3a/num_RUs)
        
        #constraint 26a
        for k in range(U_ur):
            temp3 = 0
            for b in range(num_RBs):
                temp4 = 0
                for i in range(num_RUs): 
                    temp4 += p_ur[i,k,b]*z_ur[(i,k,b)]*g_ur[i,k,b,num_slots_per_frame-1]
                temp3 += rb_bandwidth*(cp.log(1+ temp4)/0.3 - z2_ur[(k,b)]*(stats.norm.ppf(1 - error_prob))/\
                                       cp.sqrt(urllc_packet_size))
            constraint.append( temp3 >= cp.inv_pos(u1))
        
        data_ur1 = 0
        for k in range(U_ur):
            data_ur1 += data_ur[frame_num,k]
        L_cu = 1/(u_cu - data_ur1)
        L_du = 1/(u_du - data_ur1)
        rho_du = data_ur1/u_du
         #constraint 26b
        for i in range(num_RUs):
            for j in range(num_DUs):
                for k in range(U_ur): 
                    temp5 = z1_ur[(i,k)]*(dis_ur[i,k]/(3*1e8)+ rho_du/(u_du - data_ur1)) + urllc_packet_size*u1 + L_cu + y[j]*L_du
                    constraint.append( temp5 <= L_ur*x_ur)
                    #constraint.append( temp5 <= L_ur)
        #constraint 27a-d
        for j in range(num_DUs):
            constraint.append(c_j[j] >= 0)
            if frame_num ==0:
                constraint.append(c_j[j] >= y[j])
                constraint.append(c_j[j] <= y[j] + 10*(1 - u2))
            else:
                constraint.append(c_j[j] >= y[j] - y1[j])
                constraint.append(c_j[j] <= y[j] - y1[j] + 10*(1 - u2))
            constraint.append(c_j[j] <= 10*u2)
        
        
        obj = 0
        #for k in range(U_em):
        #    obj += z_em[0][k][b]
        #for k in range(U_ur):
        #    obj += z_ur[0][k][b]
        #for k in range(U_mm):
        #    obj += z_mm[0][k][b]
        objective= cp.Maximize(obj)

        # Solve problem
        prob = cp.Problem(objective, constraint)
        prob.solve(solver = 'MOSEK',verbose = True) 
        output = prob.status

    except cp.SolverError:
        output = 'non_feasible'
    print('Output is:', output)
    z1_ur0 = np.zeros((num_RUs,U_ur))
    for k in range(U_ur):
        for i in range(num_RUs):
            z1_ur0[i,k] = z1_ur[(i,k)].value
    z2_ur0 = np.zeros((U_ur,num_RBs))
    for k in range(U_ur):
        for b in range(num_RBs):
            z2_ur0[k,b] = z2_ur[(k,b)].value
    y0 = np.zeros((num_DUs))
    for j in range(num_DUs):
        y0[j] = y[j].value
    u2_0 = u2.value
    x_em0 = x_em.value
    x_ur0 = x_ur.value
    x_mm0 = x_mm.value
    
    return  x_em0,x_ur0,x_mm0, z1_ur0,z2_ur0, y0, u2_0
    #return z_em0, z_ur0, z_mm0
    
def long_sub_BB(p_em, g_em, p_ur, g_ur, p_mm, g_mm, data_ur, dis_ur, frame_num, y1,dis_em_m, dis_ur_m, dis_mm_m):
    try:
        z_em, z_ur, z_mm = {}, {}, {}
        for i in range(num_RUs):
           for b in range(num_RBs):
                for k in range(U_em):
                    z_em[(i,k,b)]= cp.Variable(integer=True)
                for k in range(U_ur):
                    z_ur[(i,k,b)]= cp.Variable(integer=True)
                for k in range(U_mm):
                    z_mm[(i,k,b)]= cp.Variable(integer=True)
        #z1_mm = {}
        z1_ur = {}
        for k in range(U_ur):
            for i in range(num_RUs):
                z1_ur[(i,k)]= cp.Variable(integer=True)
        z2_ur = {}
        for k in range(U_ur):
            for b in range(num_RBs):
                z2_ur[(k,b)]= cp.Variable(integer=True)
        #for k in range(U_mm):
        #    z1_mm[(k,frame_num)]= cp.Variable() 
        x_em, x_ur, x_mm = {}, {}, {}
        x_em =  cp.Variable(integer=True)
        x_ur =  cp.Variable(integer=True)
        x_mm =  cp.Variable(integer=True)
        y = {}
        for j in range(num_DUs):
        #    y_em[(j,frame_num)] =  cp.Variable()
            y[j] =  cp.Variable(integer=True)
        #    y_mm[(j,frame_num)] =  cp.Variable()

        c_j = {}
        for j in range(num_DUs):
            c_j[j] = cp.Variable()
        u1 = {}
        u1 =  cp.Variable()
        u2 = {}
        u2 = cp.Variable(integer=True)

        constraint= []


        constraint.append(x_em >= 0)
        constraint.append(x_em <= 1)
        constraint.append(x_ur >= 0)
        constraint.append(x_ur <= 1)
        constraint.append(x_mm >= 0)
        constraint.append(x_mm <= 1)
        
        #constraint 23g,h
        for i in range(num_RUs):
            for b in range(num_RBs): 
                for k in range(U_em):
                    constraint.append(z_em[(i,k,b)] >= 0)
                    constraint.append(z_em[(i,k,b)] <= 1)
                for k in range(U_ur):
                    constraint.append(z_ur[(i,k,b)] >= 0)
                    constraint.append(z_ur[(i,k,b)] <= 1)
                #for k in range(U_mm):
                  #  constraint.append(z_mm[(i,k,b)] >= 0)
                 #   constraint.append(z_mm[(i,k,b)] <= 1)
        
        for k in range(U_ur):
            for i in range(num_RUs):
                constraint.append(z1_ur[(i,k)] >= 0)
                constraint.append(z1_ur[(i,k)] <= 1)
        
        for k in range(U_ur):
            for b in range(num_RBs):
                constraint.append(z2_ur[(k,b)] >= 0)
                constraint.append(z2_ur[(k,b)] <= 1)

        for j in range(num_DUs):
            constraint.append(y[j] >= 0)
            constraint.append(y[j] <= 1)
        
        
        constraint.append(u1 >= 1e-4)
        constraint.append(u2 >= 0)
        constraint.append(u2 <= 1)

        #constraint 23a
        temp1 = 0
        for j in range(num_DUs):
            temp1 += y[j]*0.05 
        constraint.append(temp1 <= 1)
        temp1b = 0
        for j in range(num_DUs):
            temp1b += y[j]
        constraint.append(temp1b >= 1)
            
        
        #constraint 23b
        #constraint.append(x_em[frame_num]*0.1 + x_ur[frame_num]*0.1 + x_mm[frame_num]*0.1 <= 1)

        #constraint.append(x_em[frame_num] + x_ur[frame_num] + x_mm[frame_num] >= 1)

        #constraint 23c
        for i in range(num_RUs):
            for b in range(num_RBs):
                temp0 = 0
                for k in range(U_em):
                    temp0 += z_em[(i,k,b)]
                for k in range(U_ur):
                    temp0 += z_ur[(i,k,b)]
                for k in range(U_mm):
                    temp0 += z_mm[(i,k,b)] 
                constraint.append(temp0 <= 1)
        
        #constraint 23d
        for b in range(num_RBs): 
            for k in range(U_em):
                for i in range(num_RUs):
                    for j in range (i,num_RUs):
                        constraint.append(z_em[(i,k,b)] == z_em[(j,k,b)])
            for k in range(U_ur):
                for i in range(num_RUs):
                    for j in range (i,num_RUs):
                        constraint.append(z_ur[(i,k,b)] == z_ur[(j,k,b)])
            for k in range(U_mm):
                for i in range(num_RUs):
                    for j in range (i,num_RUs):
                        constraint.append(z_mm[(i,k,b)] == z_mm[(j,k,b)])

        
        for k in range(U_ur):
            for i in range(num_RUs):
                temp1a = 0
                for b in range(num_RBs): 
                    temp1a += z_ur[(i,k,b)]
                constraint.append(temp1a/num_RBs <= z1_ur[(i,k)])

        
        for k in range(U_ur):
            for i in range(num_RUs):
                temp2a = 0
                for b in range(num_RBs): 
                    temp2a += z_ur[(i,k,b)]
                constraint.append(z1_ur[(i,k)] - temp2a/(num_RBs) <= 0.99999)
        
        for k in range(U_ur):
            for b in range(num_RBs):
                temp3a = 0
                for i in range(num_RUs):
                    temp3a += z_ur[(i,k,b)]
                constraint.append(z2_ur[(k,b)] == temp3a/num_RUs)
        
        data_ur1 = 0
        for k in range(U_ur):
            data_ur1 += data_ur[frame_num,k]
        L_cu = 1/(u_cu - data_ur1)
        L_du = 1/(u_du - data_ur1)
        rho_du = data_ur1/u_du

        #constraint 26b
        for i in range(num_RUs):
            for j in range(num_DUs):
                for k in range(U_ur): 
                    #temp5 = z_ur[(i,k,b)]*(dis_ur[i,k]/(3*1e8)+ rho_du/(u_du - data_ur1)) + urllc_packet_size*u1 + L_cu + y[j]*L_du
                    temp5 = z1_ur[(i,k)]*(dis_ur[i,k]/(3*1e8)+ rho_du/(u_du - data_ur1)) + urllc_packet_size*u1 + L_cu + y[j]*L_du
                    constraint.append( temp5 <= L_ur*x_ur)
                    #constraint.append( temp5 <= L_ur)
        
        #constraint 26a

        for k in range(U_ur):
            temp3 = 0
            for b in range(num_RBs):
                temp4 = 0
                for i in range(num_RUs): 
                    temp4 += p_ur[i,k,b]*z_ur[(i,k,b)]*g_ur[i,k,b,num_slots_per_frame-1]
                temp3 += rb_bandwidth*(cp.log(1+ temp4)/0.3 - z2_ur[(k,b)]*(stats.norm.ppf(1 - error_prob))/\
                                       cp.sqrt(urllc_packet_size))
            constraint.append( temp3 >= cp.inv_pos(u1))
        
        
        #constraint 28b
        for k in range(U_em):
            temp6 = 0
            for b in range(num_RBs):
                temp7 = 0
                for i in range(num_RUs): 
                    temp7 += p_em[i,k,b]*z_em[(i,k,b)]*g_em[i,k,b,num_slots_per_frame-1]
                temp6 += rb_bandwidth*(cp.log(1+ temp7)/0.3)
            constraint.append(temp6 >= R_em*x_em)
            #constraint.append(temp6 >= R_em)
        
        
        for k in range(U_mm):
            for i in dis_mm_m[k]:
                temp9 = 0
                for b in range(num_RBs):
                    temp9 += p_mm[i,k,b]*z_mm[(i,k,b)]*g_mm[i,k,b,num_slots_per_frame-1]
                constraint.append( temp9 >= gamma_mm*x_mm)
                #constraint.append( temp9 >= gamma_mm)
        for k in range(U_ur):
            for i in dis_ur_m[k]:
                temp9a = 0
                for b in range(num_RBs):
                    temp9a += p_ur[i,k,b]*z_ur[(i,k,b)]*g_ur[i,k,b,num_slots_per_frame-1]
                constraint.append( temp9a >= gamma_mm*x_ur)
                #constraint.append( temp9a >= gamma_mm)
        
        for k in range(U_em):
            for i in dis_em_m[k]:
                temp9b = 0
                for b in range(num_RBs):
                    temp9b += p_em[i,k,b]*z_em[(i,k,b)]*g_em[i,k,b,num_slots_per_frame-1]
                constraint.append( temp9b >= gamma_mm*x_em)
                #constraint.append( temp9b >= gamma_mm)
        
        #constraint 27a-d
        for j in range(num_DUs):
            constraint.append(c_j[j] >= 0)
            if frame_num ==0:
                constraint.append(c_j[j] >= y[j])
                constraint.append(c_j[j] <= y[j] + 10*(1 - u2))
            else:
                constraint.append(c_j[j] >= y[j] - y1[j])
                constraint.append(c_j[j] <= y[j] - y1[j] + 10*(1 - u2))
            constraint.append(c_j[j] <= 10*u2)
        
        temp10 = 0
        for j in range(num_DUs):
            temp10 += c_j[j]
        
        temp11 = 0
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(U_em):
                    temp11 += z_em[(i,k,b)]
                for k in range(U_ur):
                    temp11 += z_ur[(i,k,b)]
                for k in range(U_mm):
                    temp11 += z_mm[(i,k,b)]
        
        #obj = phi *(x_em[frame_num] + x_ur[frame_num] + x_mm[frame_num]) - eta*temp10
        obj = alpha*(11.00*x_em + 12.00*x_ur + 60.00*x_mm) - beta*(temp10 + temp11)
        #obj = 10000*(30*x_em + 20*x_ur + 40*x_mm) - (temp10 + temp11)
        #obj = 0
        objective= cp.Maximize(obj)

        # Solve problem
        prob = cp.Problem(objective, constraint)
        #prob.solve(solver = 'SCIP',verbose = True) 
        prob.solve(solver = 'MOSEK',verbose = True)
        output = prob.status

    except cp.SolverError:
        output = 'non_feasible'
    print('Output is:', output)
    print('value u1:',u1.value)
    print('object value u1:',prob.value)
    print('x_em:',x_em.value)
    print('x_ur:',x_ur.value)
    print('x_mm:',x_mm.value)
    
    z_em1 = np.zeros((num_RUs,U_em,num_RBs))
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_em):
                #for t in range(num_slots_per_frame):
                z_em1[i,k,b] = math.floor(z_em[(i,k,b)].value +0.5)
                #z_em1[i,k,b] = z_em[(i,k,b)].value
    z_ur1 = np.zeros((num_RUs,U_ur,num_RBs))
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_ur):
                z_ur1[i,k,b] = math.floor(z_ur[(i,k,b)].value +0.5)
                #z_ur1[i,k,b] = z_ur[(i,k,b)].value
    z_mm1 = np.zeros((num_RUs,U_mm,num_RBs))
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_mm):
                z_mm1[i,k,b] = math.floor(z_mm[(i,k,b)].value +0.5)
                #z_mm1[i,k,b] = z_mm[(i,k,b)].value
    z1_ur1 = np.zeros((num_RUs,U_ur))
    for k in range(U_ur):
        for i in range(num_RUs):
            z1_ur1[i,k] = math.floor(z1_ur[(i,k)].value +0.5)
    z2_ur1 = np.zeros((U_ur,num_RBs))
    for k in range(U_ur):
        for b in range(num_RBs):
            z2_ur1[k,b] = math.floor(z2_ur[(k,b)].value +0.5)
    y1 = np.zeros((num_DUs))
    for j in range(num_DUs):
        y1[j] = math.floor(y[j].value+0.5)
    

    for k in range(U_em):
        filename = fr"bb/z_em{k}.csv"
        np.savetxt(filename, z_em1[:,k,:], delimiter=',')
    for k in range(U_ur):
        filename = fr"bb/z_ur{k}.csv"
        np.savetxt(filename, z_ur1[:,k,:], delimiter=',')
    for k in range(U_mm):
        filename = fr"bb/z_mm{k}.csv"
        np.savetxt(filename, z_mm1[:,k,:], delimiter=',')
    filename = fr"bb/z1_ur.csv"
    np.savetxt(filename, z1_ur1[:], delimiter=',')
    filename = fr"bb/y.csv"
    np.savetxt(filename, y1[:], delimiter=',')
    return z_em1, z_ur1, z_mm1, z1_ur1, z2_ur1, y1, math.floor(x_em.value +0.5), math.floor(x_ur.value+0.5), math.floor(x_mm.value+0.5), prob.value

def env(z_em,z_ur,z_mm,p_em,p_ur,p_mm,dis_em,dis_mm,dis_ur,g_em,g_ur,g_mm,x_em,x_ur,x_mm,y,dis_em_m,dis_ur_m,dis_mm_m,data_ur,frame_num):
    Reward = 0
    Flag = 1
    #check constraint
    temp0 = 0
    for j in range(num_DUs):
        temp0 += y[j]*0.05 
    if temp0 <= 1:
        Reward += 1
    else:
        Reward -= 2
        Flag *= 0
        print('Constraint 23a is not met!!!!')

    for i in range(num_RUs):
        for b in range(num_RBs):
            temp1 = 0
            for k in range(U_em):
                temp1 += z_em[i,k,b]
            for k in range(U_ur):
                temp1 += z_ur[i,k,b]
            #for k in range(U_mm):
            #    temp1 += z_mm[i,k,b] 
            if temp1 > 1: 
                Reward -= 2
                print('Constraint 23c is not met!!!!')
                Flag *= 0
            else:
                Reward += 1
    p_ru = []
    for i in range(num_RUs):
        temp2 = 0
        for b in range(num_RBs):
            for k in range(U_em):  
                temp2 += p_em[i,k,b]*z_em[i,k,b]
            for k in range(U_ur):  
                temp2 += p_ur[i,k,b]*z_ur[i,k,b]
            for k in range(U_mm):  
                temp2 += p_mm[i,k,b]*z_mm[i,k,b]
        p_ru.append(temp2)
        if abs(temp2 - max_tx_power_watts) <= 0.0001:
            Reward += 1
        elif temp2 <= max_tx_power_watts:
            Reward += 1
        else:
            print('Constraint 24a (total power) is not met!!!!')
            print(max_tx_power_watts)
            print(temp2)
            Reward -= 2
            Flag *= 0
    R_user = []
    for k in range(U_em):
        temp3 = 0
        for b in range(num_RBs):
            temp4 = 0
            for i in range(num_RUs): 
                temp4 += p_em[i,k,b]*z_em[i,k,b]*g_em[i,k,b,num_slots_per_frame-1]
            temp3 += rb_bandwidth*(np.log2(1+ temp4))
        R_user.append(temp3)
        if temp3 >= R_em*x_em:
            Reward += 1
        else:
            print('Constraint 28b (throughput) is not met!!!!')
            print(R_em)
            print(temp3)
            Reward -= 2
            Flag *= 0
    '''
    for k in range(U_mm):
        for i in dis_mm_m[k]:
            temp5 = 0
            for b in range(num_RBs):
                temp5 += p_mm[i,k,b]*z_mm[i,k,b]*g_mm[i,k,b,num_slots_per_frame-1]
            if temp5 >= gamma_mm*x_mm:
                Reward += 1
            else:
                Reward -= 2
                print('Constraint SNR mm is not met!!!!')
                Flag *= 0
    '''
    for k in range(U_ur):
        for i in dis_ur_m[k]:
            temp5a = 0
            for b in range(num_RBs):
                temp5a += p_ur[i,k,b]*z_ur[i,k,b]*g_ur[i,k,b,num_slots_per_frame-1]
            if temp5a >= gamma_mm*x_ur:
                Reward += 1
            else:
                print('Constraint SNR ur is not met!!!!')
                Reward -= 2
                Flag *= 0
    '''
    for k in range(U_em):
        for i in dis_em_m[k]:
            temp5b = 0
            for b in range(num_RBs):
                temp5b += p_em[i,k,b]*z_em[i,k,b]*g_em[i,k,b,num_slots_per_frame-1]
            if temp5b >= gamma_mm*x_em:
                Reward += 1
            else:
                print('Constraint SNR em is not met!!!!')
                Reward -= 2
                Flag *= 0
    '''
    data_ur1 = 0
    for k in range(U_ur):
        data_ur1 += data_ur[frame_num,k]
    L_cu = 1/(u_cu - data_ur1)
    L_du = 1/(u_du - data_ur1)
    rho_du = data_ur1/u_du

    z2_ur = np.zeros((U_ur,num_RBs))
    for k in range(U_ur):
        for b in range(num_RBs):
            temp6 = 0
            for i in range(num_RUs):
                temp6 += z_ur[i,k,b]
            z2_ur[k,b] = temp6/num_RUs
    R_ur = np.zeros(U_ur)
    for k in range(U_ur):
        temp7 = 0
        for b in range(num_RBs):
            temp8 = 0
            for i in range(num_RUs): 
                temp8 += p_ur[i,k,b]*z_ur[i,k,b]*g_ur[i,k,b,num_slots_per_frame-1]
            temp7 += rb_bandwidth*(np.log2(1+ temp8) - z2_ur[k,b]*(stats.norm.ppf(1 - error_prob))/np.sqrt(urllc_packet_size))
        R_ur[k] = temp7

    z1_ur = np.zeros((num_RUs,U_ur))
    for k in range(U_ur):
        for i in range(num_RUs):
            temp9 = 0
            for b in range(num_RBs):
                temp9 += z_ur[i,k,b]
            if temp9 >= 1:
                z1_ur[i,k] = 1
#E2E
    temp10 = np.zeros((num_DUs,num_RUs,U_ur))
    L_user = []
    for k in range(U_ur):
        for i in range(num_RUs):
            for j in range(num_DUs): 
                if R_ur[k] >0:
                    temp10[j,i,k] = z1_ur[i,k]*(dis_ur[i,k]/(3*1e8)+ rho_du/(u_du - data_ur1)) + urllc_packet_size/R_ur[k] + L_cu + y[j]*L_du
        L_user.append(np.max(temp10[:,:,k]))
        if np.max(temp10[:,:,k])<= L_ur*x_ur and np.max(temp10[:,:,k]) > 0:
            Reward += 1
        else:
            print('Constraint 28c (latency) is not met!!!!')
            Reward -= 2
            Flag *= 0
    
    return Flag, R_user, L_user, p_ru

def benchmark(z_em, z_ur, z_mm, p_em, g_em, p_ur, g_ur, p_mm, g_mm, data_ur, dis_ur, frame_num):
    try:
        z1_ur = {}
        for k in range(U_ur):
            for i in range(num_RUs):
                z1_ur[(i,k)]= cp.Variable(integer=True)
        z2_ur = {}
        for k in range(U_ur):
            for b in range(num_RBs):
                z2_ur[(k,b)]= cp.Variable(integer=True)
        
        y = {}
        for j in range(num_DUs):
            y[j] =  cp.Variable(integer=True)

        x_em, x_ur, x_mm = {}, {}, {}
        x_em =  cp.Variable(integer=True)
        x_ur =  cp.Variable(integer=True)
        x_mm =  cp.Variable(integer=True)

        c_j = {}
        for j in range(num_DUs):
            c_j[j] = cp.Variable()
        
        u2 = {}
        u2 = cp.Variable(integer=True)

        u1 = {}
        u1 =  cp.Variable()

        constraint= []
        
        for k in range(U_ur):
            for i in range(num_RUs):
                constraint.append(z1_ur[(i,k)] >= 0)
                constraint.append(z1_ur[(i,k)] <= 1)
        
        for k in range(U_ur):
            for b in range(num_RBs):
                constraint.append(z2_ur[(k,b)] >= 0)
                constraint.append(z2_ur[(k,b)] <= 1)


        for j in range(num_DUs):
            constraint.append(y[j] >= 0)
            constraint.append(y[j] <= 1)
        
        constraint.append(x_em >= 0)
        constraint.append(x_em <= 1)
        constraint.append(x_ur >= 0)
        constraint.append(x_ur <= 1)
        constraint.append(x_mm >= 0)
        constraint.append(x_mm <= 1)

        constraint.append(u1 >= 1e-4)
        constraint.append(u2 >= 0)
        constraint.append(u2 <= 1)

            
        temp1 = 0
        for j in range(num_DUs):
            temp1 += y[j]*0.05 
        constraint.append(temp1 <= 1)

        
        for k in range(U_ur):
            for i in range(num_RUs):
                temp1a = 0
                for b in range(num_RBs): 
                    temp1a += z_ur[i,k,b]
                constraint.append(temp1a/num_RBs <= z1_ur[(i,k)])

        
        for k in range(U_ur):
            for i in range(num_RUs):
                temp2a = 0
                for b in range(num_RBs): 
                    temp2a += z_ur[i,k,b]
                constraint.append(z1_ur[(i,k)] - temp2a/(num_RBs) <= 0.99999)
        
        for k in range(U_ur):
            for b in range(num_RBs):
                temp3a = 0
                for i in range(num_RUs):
                    temp3a += z_ur[i,k,b]
                constraint.append(z2_ur[(k,b)] == temp3a/num_RUs)
        
        #constraint 26a
        for k in range(U_ur):
            temp3 = 0
            for b in range(num_RBs):
                temp4 = 0
                for i in range(num_RUs): 
                    temp4 += p_ur[i,k,b]*z_ur[i,k,b]*g_ur[i,k,b,num_slots_per_frame-1]
                temp3 += rb_bandwidth*(cp.log(1+ temp4)/0.3 - z2_ur[(k,b)]*(stats.norm.ppf(1 - error_prob))/\
                                       cp.sqrt(urllc_packet_size))
            constraint.append( temp3 >= cp.inv_pos(u1))
        
        data_ur1 = 0
        for k in range(U_ur):
            data_ur1 += data_ur[frame_num,k]
        L_cu = 1/(u_cu - data_ur1)
        L_du = 1/(u_du - data_ur1)
        rho_du = data_ur1/u_du
         #constraint 26b
        for i in range(num_RUs):
            for j in range(num_DUs):
                for k in range(U_ur): 
                    temp5 = z1_ur[(i,k)]*(dis_ur[i,k]/(3*1e8)+ rho_du/(u_du - data_ur1)) + urllc_packet_size*u1 + L_cu + y[j]*L_du
                    constraint.append( temp5 <= L_ur*x_ur)
                    #constraint.append( temp5 <= L_ur)
        
        #constraint 28b
        for k in range(U_em):
            temp6 = 0
            for b in range(num_RBs):
                temp7 = 0
                for i in range(num_RUs): 
                    temp7 += p_em[i,k,b]*z_em[i,k,b]*g_em[i,k,b,num_slots_per_frame-1]
                temp6 += rb_bandwidth*(cp.log(1+ temp7)/0.3)
            constraint.append(temp6 >= R_em*x_em)
            #constraint.append(temp6 >= R_em)
        
        
        for k in range(U_mm):
            for i in dis_mm_m[k]:
                temp9 = 0
                for b in range(num_RBs):
                    temp9 += p_mm[i,k,b]*z_mm[i,k,b]*g_mm[i,k,b,num_slots_per_frame-1]
                constraint.append( temp9 >= gamma_mm*x_mm)
        
        '''
        for k in range(U_em):
            for i in dis_em_m[k]:
                temp9a = 0
                for b in range(num_RBs):
                    temp9a += p_em[i,k,b]*z_em[i,k,b]*g_em[i,k,b,num_slots_per_frame-1]
                constraint.append( temp9a >= gamma_mm*x_em)
        '''

        for k in range(U_ur):
            for i in dis_ur_m[k]:
                temp9b = 0
                for b in range(num_RBs):
                    temp9b += p_ur[i,k,b]*z_ur[i,k,b]*g_ur[i,k,b,num_slots_per_frame-1]
                constraint.append( temp9b >= gamma_mm*x_ur)
        
        #constraint 27a-d
        for j in range(num_DUs):
            constraint.append(c_j[j] >= 0)
            if frame_num ==0:
                constraint.append(c_j[j] >= y[j])
                constraint.append(c_j[j] <= y[j] + 10*(1 - u2))
            constraint.append(c_j[j] <= 10*u2)
        
        
        temp10 = 0
        for j in range(num_DUs):
            temp10 += c_j[j]
        
        temp11 = 0
        for i in range(num_RUs):
            for b in range(num_RBs):
                for k in range(U_em):
                    temp11 += z_em[i,k,b]
                for k in range(U_ur):
                    temp11 += z_ur[i,k,b]
                for k in range(U_mm):
                    temp11 += z_mm[i,k,b]
        
        #obj = phi *(x_em[frame_num] + x_ur[frame_num] + x_mm[frame_num]) - eta*temp10
        obj = alpha*(11.00*x_em + 12.00*x_ur + 60.00*x_mm) - beta*(temp10 + temp11)
        #obj = 10000*(30*x_em + 20*x_ur + 40*x_mm) - (temp10 + temp11)
        #obj = 0
        objective= cp.Maximize(obj)

        # Solve problem
        prob = cp.Problem(objective, constraint)
        prob.solve(solver = 'MOSEK',verbose = True) 
        output = prob.status

    except cp.SolverError:
        output = 'non_feasible'
    print('Output is:', output)
    z1_ur0 = np.zeros((num_RUs,U_ur))
    for k in range(U_ur):
        for i in range(num_RUs):
            z1_ur0[i,k] = math.floor(z1_ur[(i,k)].value + 0.5)
    z2_ur0 = np.zeros((U_ur,num_RBs))
    for k in range(U_ur):
        for b in range(num_RBs):
            z2_ur0[k,b] = math.floor(z2_ur[(k,b)].value + 0.5)
    y0 = np.zeros((num_DUs))
    for j in range(num_DUs):
        y0[j] = math.floor(y[j].value + 0.5)
    
    u2_0 = math.floor(u2.value + 0.5)
    x_em0 = math.floor(x_em.value + 0.5)
    x_ur0 = math.floor(x_ur.value + 0.5)
    x_mm0 = math.floor(x_mm.value + 0.5)
    temp10 = 0
    for j in range(num_DUs):
        temp10 += c_j[j].value
        
    temp11 = 0
    for i in range(num_RUs):
        for b in range(num_RBs):
            for k in range(U_em):
                temp11 += z_em[i,k,b]
            for k in range(U_ur):
                temp11 += z_ur[i,k,b]
            for k in range(U_mm):
                temp11 += z_mm[i,k,b]
    op_value = alpha*(11.00*math.floor(x_em.value+0.5) + 12.00*math.floor(x_ur.value+0.5) + 60.00*math.floor(x_mm.value+0.5)) - beta*(temp10 + temp11)
    print('x_em_bm:', x_em.value)
    print('x_ur_bm:', x_ur.value)
    print('x_mm_bm:', x_mm.value)
    filename = fr"ben/z1_ur.csv"
    np.savetxt(filename, z1_ur0[:], delimiter=',')
    filename = fr"ben/y.csv"
    np.savetxt(filename, y0[:], delimiter=',')
    return  x_em0,x_ur0,x_mm0, z1_ur0,z2_ur0, y0, u2_0, op_value

dis_ur, dis_em, dis_mm = generate_position1()
g_em, g_ur, g_mm = channel_gain1(dis_em, dis_ur, dis_mm)
data_ur, data_em, data_mm = generate_lambda()

dis_ur_m = np.zeros((U_ur, 3), dtype=int)
dis_mm_m = np.zeros((U_mm, 3), dtype=int)
dis_em_m = np.zeros((U_em, 3), dtype=int)
for k in range(U_mm):
    dis_mm_m[k] = np.argsort(dis_mm[:, k])[:3]
for k in range(U_em):
    dis_em_m[k] = np.argsort(dis_em[:, k])[:3]
for k in range(U_ur):
    dis_ur_m[k] = np.argsort(dis_ur[:, k])[:3]

p_em = np.zeros((num_RUs,U_em,num_RBs))
p_ur = np.zeros((num_RUs,U_ur,num_RBs))
p_mm = np.zeros((num_RUs,U_mm,num_RBs))
pp = max_tx_power_watts/(num_RBs)
for i in range(num_RUs):
    p_em[i,:,:] = pp*np.ones((U_em,num_RBs))
    p_ur[i,:,:] = pp*np.ones((U_ur,num_RBs))
    p_mm[i,:,:] = pp*np.ones((U_mm,num_RBs))

print('max g_ur:',np.max(g_ur))
print('min g_ur:',np.min(g_ur))
print('mean g_ur:',np.mean(g_ur))
print('max g_em:',np.max(g_em))
print('min g_em:',np.min(g_em))
print('mean g_em:',np.mean(g_em))
print('max g_mm:',np.max(g_mm))
print('min g_mm:',np.min(g_mm))
print('mean g_mm:',np.mean(g_mm))

############## BB #################

y_temp = np.zeros(num_DUs)
z_em, z_ur, z_mm, z1_ur,z2_ur, y, x_em, x_ur, x_mm, op_value = long_sub_BB(p_em, g_em, p_ur, g_ur, p_mm, g_mm, data_ur, dis_ur, 0, y_temp,dis_em_m, dis_ur_m, dis_mm_m)
Flag, R_bb, L_bb, p_ru_bb = env(z_em,z_ur,z_mm,p_em,p_ur,p_mm,dis_em,dis_mm,dis_ur,g_em,g_ur,g_mm,x_em,x_ur,x_mm,y,dis_em_m,dis_ur_m,dis_mm_m,data_ur,0)
print(Flag)
'''
R_bb = []
for k in range(U_em):
    temp = 0
    for b in range(num_RBs):
        temp1 = 0
        for i in range(num_RUs): 
            temp1 += p_em[i,k,b]*z_em[i,k,b]*g_em[i,k,b,num_slots_per_frame-1]
        temp += rb_bandwidth*(np.log(1+ temp1)/0.3)
    R_bb.append(temp)
'''
num_RBs_BB = []
temp_em = 0
for i in range (num_RUs):
    for b in range(num_RBs):
        for k in range(U_em):
            temp_em += z_em[i,k,b]
num_RBs_BB.append(temp_em/num_RUs)
temp_mm = 0
for i in range (num_RUs):
    for b in range(num_RBs):
        for k in range(U_mm):
            temp_mm += z_mm[i,k,b]
num_RBs_BB.append(temp_mm/num_RUs)
temp_ur = 0
for i in range (num_RUs):
    for b in range(num_RBs):
        for k in range(U_ur):
            temp_ur += z_ur[i,k,b]
num_RBs_BB.append(temp_ur/num_RUs)

#R_bb = []
#p_em, p_ur, p_mm,R_bb_s, L_bb, mean_p_bb, p_ru_bb = short_subproblem(dis_ur, y ,z_em, z_ur, z_mm, g_em, g_ur, g_mm, data_ur, 0, z2_ur, dis_mm_m,dis_ur_m, 0)
#print(R_bb_s)

############### Benchmark 1####################
p_em = np.zeros((num_RUs,U_em,num_RBs))
p_ur = np.zeros((num_RUs,U_ur,num_RBs))
p_mm = np.zeros((num_RUs,U_mm,num_RBs))
pp = max_tx_power_watts/((num_RBs))
for i in range(num_RUs):
    p_em[i,:,:] = pp*np.ones((U_em,num_RBs))
    p_ur[i,:,:] = pp*np.ones((U_ur,num_RBs))
    p_mm[i,:,:] = pp*np.ones((U_mm,num_RBs))

def number(num_RBs,num_users):
    if num_RBs%num_users == 0:
        a = num_RBs//num_users
        b = num_RBs//num_users
    else:
        a = num_RBs//num_users + 1
        b = num_RBs//num_users
    x = num_RBs - b*num_users
    return a,b,x

def randomize_RBs(num_users, num_RBs, a,b,x):
    RB_assignment1 = np.zeros(num_RBs, dtype=int)
    RB_assignment = np.zeros((num_users, num_RBs), dtype=int)
    for user in range(num_users-x):
        # Determine the number of RBs this user will use
        num_RBs_to_use = b
        # Randomly select non-overlapping RBs for this user
        available_RBs = np.where(RB_assignment1 == 0)[0]
        selected_RBs = np.random.choice(available_RBs, size=num_RBs_to_use, replace=False)
        # Assign 1s to the selected RBs
        RB_assignment1[selected_RBs] = 1
        RB_assignment[user, selected_RBs] = 1
    for user in range(num_users-x,num_users):
        num_RBs_to_use = a
        # Randomly select non-overlapping RBs for this user
        available_RBs = np.where(RB_assignment1 == 0)[0]
        selected_RBs = np.random.choice(available_RBs, size=num_RBs_to_use, replace=False)
        # Assign 1s to the selected RBs
        RB_assignment1[selected_RBs] = 1
        RB_assignment[user, selected_RBs] = 1
    return RB_assignment

a,b,x = number(num_RBs, U)
z_total = randomize_RBs(U, num_RBs, a,b,x)

z_em_bm = np.zeros((num_RUs,U_em,num_RBs))
z_ur_bm = np.zeros((num_RUs,U_ur,num_RBs))
z_mm_bm = np.zeros((num_RUs,U_mm,num_RBs))

for i in range(num_RUs):
    for k in range(U_em):
        z_em_bm[i,k,:] = z_total[k,:]
    for k in range(U_ur):
        z_ur_bm[i,k,:] = z_total[k+U_em,:]
    for k in range(U_mm):
        z_mm_bm[i,k,:] = z_total[k+U_em+U_ur,:]

for k in range(U_em):
    filename = fr"ben/z_em{k}.csv"
    np.savetxt(filename, z_em_bm[:,k,:], delimiter=',')
for k in range(U_ur):
    filename = fr"ben/z_ur{k}.csv"
    np.savetxt(filename, z_ur_bm[:,k,:], delimiter=',')
for k in range(U_mm):
    filename = fr"ben/z_mm{k}.csv"
    np.savetxt(filename, z_mm_bm[:,k,:], delimiter=',')

x_em_bm,x_ur_bm,x_mm_bm,z1_ur_bm,z2_ur_bm,y_bm, u2_bm,op_value_bm = benchmark(z_em_bm, z_ur_bm, z_mm_bm, p_em, g_em, p_ur, g_ur, p_mm, g_mm, data_ur, dis_ur, 0)
y_bm = np.ones(3)
Flag, R_ben, L_ben, p_ru_ben = env(z_em_bm,z_ur_bm,z_mm_bm,p_em,p_ur,p_mm,dis_em,dis_mm,dis_ur,g_em,g_ur,g_mm,x_em_bm,x_ur_bm,x_mm_bm,y_bm,dis_em_m,dis_ur_m,dis_mm_m,data_ur,0)
print(Flag)
'''
R_ben = []
for k in range(U_em):
    temp = 0
    for b in range(num_RBs):
        temp1 = 0
        for i in range(num_RUs): 
            temp1 += p_em[i,k,b]*z_em_bm[i,k,b]*g_em[i,k,b,num_slots_per_frame-1]
        temp += rb_bandwidth*(np.log(1+ temp1)/0.3)
    R_ben.append(temp)
'''
num_RBs_Ben = []
temp_em = 0
for i in range (num_RUs):
    for b in range(num_RBs):
        for k in range(U_em):
            temp_em += z_em_bm[i,k,b]
num_RBs_Ben.append(temp_em/num_RUs)
temp_mm = 0
for i in range (num_RUs):
    for b in range(num_RBs):
        for k in range(U_mm):
            temp_mm += z_mm_bm[i,k,b]
num_RBs_Ben.append(temp_mm/num_RUs)
temp_ur = 0
for i in range (num_RUs):
    for b in range(num_RBs):
        for k in range(U_ur):
            temp_ur += z_ur_bm[i,k,b]
num_RBs_Ben.append(temp_ur/num_RUs)

#p_em, p_ur, p_mm, R_ben_s, L_ben, mean_p_ben, p_ru_ben = short_subproblem(dis_ur, y_bm ,z_em_bm, z_ur_bm, z_mm_bm, g_em, g_ur, g_mm, data_ur, 0, z2_ur_bm, dis_mm_m,dis_ur_m, 0)

############### SCA ####################

p_em = np.zeros((num_RUs,U_em,num_RBs))
p_ur = np.zeros((num_RUs,U_ur,num_RBs))
p_mm = np.zeros((num_RUs,U_mm,num_RBs))
pp = max_tx_power_watts/((num_RBs))
for i in range(num_RUs):
    p_em[i,:,:] = pp*np.ones((U_em,num_RBs))
    p_ur[i,:,:] = pp*np.ones((U_ur,num_RBs))
    p_mm[i,:,:] = pp*np.ones((U_mm,num_RBs))

y_temp = np.zeros((num_DUs))
z_em0 = np.ones((num_RUs,U_em, num_RBs))
z_ur0 = np.ones((num_RUs,U_ur, num_RBs,))
z_mm0 = np.ones((num_RUs,U_mm, num_RBs))
x_em0, x_ur0, x_mm0, z1_ur0,z2_ur0, y0, u2_0 = long_sub_SCA_0(z_em0, z_ur0, z_mm0, p_em, g_em, p_ur, g_ur, p_mm, g_mm, data_ur, dis_ur, 0, y_temp)
e = True
iteration = 0
previous_optimal_value = 0
convergence = 0
#z_em1, z_ur1, z_mm1, z1_ur1, y1, u2_1, x_em1,x_ur1,x_mm1,optimal_value = long_sub_SCA(p_em, g_em, p_ur, g_ur, p_mm, g_mm, data_ur, dis_ur, 0, y_temp, z_em0, z_ur0, z_mm0, z1_ur0, y0, x_em0,x_ur0,x_mm0,u2_0)
value_sca = []
value_bb = []
value_ben = []
num_RBs_SCA = []
while e:
    z_em1, z_ur1, z_mm1, z1_ur1,z2_ur1, y1, u2_1, x_em1,x_ur1,x_mm1,optimal_value,op_value_sca = long_sub_SCA(p_em, g_em, p_ur, g_ur, p_mm, g_mm, data_ur, dis_ur, 0, y_temp, z_em0, z_ur0, z_mm0, z1_ur0,z2_ur0, y0, x_em0,x_ur0,x_mm0,u2_0,dis_em_m, dis_ur_m, dis_mm_m)
    #value_sca.append(op_value_sca)
    value_sca.append(optimal_value)
    value_bb.append(op_value)
    value_ben.append(op_value_bm)
    convergence = abs(optimal_value - previous_optimal_value)
    print('value:', convergence)
    z_em0, z_ur0, z_mm0, z1_ur0,z2_ur0, y0, u2_0, x_em0,x_ur0,x_mm0, previous_optimal_value = z_em1, z_ur1, z_mm1, z1_ur1,z2_ur1, y1, u2_1, x_em1,x_ur1,x_mm1,optimal_value
    if convergence <= 0.01:
        e = False
        for b in range(num_RBs):
            for k in range(U_em):
                for i in range(num_RUs):
                    z_em1[i,k,b] = math.floor(z_em0[i,k,b]+0.5)
        
        for b in range(num_RBs):
            for k in range(U_ur):
                for i in range(num_RUs):
                    z_ur1[i,k,b] = math.floor(z_ur0[i,k,b]+0.5)

        for b in range(num_RBs):
            for k in range(U_mm):
                for i in range(num_RUs):
                    z_mm1[i,k,b] = math.floor(z_mm0[i,k,b]+0.5)
        
        for j in range(num_DUs):
            y1[j] = math.floor(y0[j]+0.5)
        x_em1 = math.floor(x_em0 +0.5)
        x_mm1 = math.floor(x_mm0 +0.5)
        x_ur1 = math.floor(x_ur0 +0.5)
        #for k in range(U_ur):
        #    z1_ur1[k] = math.floor(z1_ur0[k]+1)
temp_em = 0
for i in range (num_RUs):
    for b in range(num_RBs):
        for k in range(U_em):
            temp_em += z_em1[i,k,b]
num_RBs_SCA.append(temp_em/num_RUs)
temp_mm = 0
for i in range (num_RUs):
    for b in range(num_RBs):
        for k in range(U_mm):
            temp_mm += z_mm1[i,k,b]
num_RBs_SCA.append(temp_mm/num_RUs)
temp_ur = 0
for i in range (num_RUs):
    for b in range(num_RBs):
        for k in range(U_ur):
            temp_ur += z_ur1[i,k,b]
num_RBs_SCA.append(temp_ur/num_RUs)

for k in range(U_em):
    filename = f"sca/z_em{k}.csv"
    np.savetxt(filename, z_em1[:,k,:], delimiter=',')
for k in range(U_ur):
    filename = f"sca/z_ur{k}.csv"
    np.savetxt(filename, z_ur1[:,k,:], delimiter=',')
for k in range(U_mm):
    filename = f"sca/z_mm{k}.csv"
    np.savetxt(filename, z_mm1[:,k,:], delimiter=',')
filename = fr"sca/y.csv"
np.savetxt(filename, y1, delimiter=',')
Flag, R_sca, L_sca, p_ru_sca = env(z_em1,z_ur1,z_mm1,p_em,p_ur,p_mm,dis_em,dis_mm,dis_ur,g_em,g_ur,g_mm,x_em1,x_ur1,x_mm1,y1,dis_em_m,dis_ur_m,dis_mm_m,data_ur,0)
print(Flag)
'''
R_sca = []
for k in range(U_em):
    temp = 0
    for b in range(num_RBs):
        temp1 = 0
        for i in range(num_RUs): 
            temp1 += p_em[i,k,b]*z_em1[i,k,b]*g_em[i,k,b,num_slots_per_frame-1]
        temp += rb_bandwidth*(np.log(1+ temp1)/0.3)
    R_sca.append(temp)
'''
p_em, p_ur, p_mm,R_sca_s,L_sca, mean_p_sca, p_ru_sca = short_subproblem(dis_ur, y1 ,z_em1, z_ur1, z_mm1, g_em, g_ur, g_mm, data_ur, 0, z2_ur1, dis_mm_m,dis_ur_m, 0)

print(num_RBs_BB)
print(num_RBs_SCA)
#print(mean_p_bb)
#print(mean_p_sca)
num_RBs_em = np.zeros(3)
num_RBs_ur = np.zeros(3)
num_RBs_mm = np.zeros(3)
num_RBs_unused = np.zeros(3)
num_RBs_em[0] = num_RBs_BB[0]
num_RBs_em[1] = num_RBs_SCA[0]
num_RBs_em[2] = num_RBs_Ben[0]
num_RBs_mm[0] = num_RBs_BB[1]
num_RBs_mm[1] = num_RBs_SCA[1]
num_RBs_mm[2] = num_RBs_Ben[1]
num_RBs_ur[0] = num_RBs_BB[2]
num_RBs_ur[1] = num_RBs_SCA[2]
num_RBs_ur[2] = num_RBs_Ben[2]
num_RBs_unused[0] = 100 - num_RBs_em[0] - num_RBs_mm[0] - num_RBs_ur[0] 
num_RBs_unused[1] = 100 - num_RBs_em[1] - num_RBs_mm[1] - num_RBs_ur[1]
num_RBs_unused[2] = 100 - num_RBs_em[2] - num_RBs_mm[2] - num_RBs_ur[2]

p_ru_used = np.zeros(3)
p_ru_unused = np.zeros(3)
p_ru_used[0] = 100*p_ru_bb[0]/max_tx_power_watts
p_ru_used[1] = 100*p_ru_sca[0]/max_tx_power_watts
p_ru_used[2] = 100*p_ru_ben[0]//max_tx_power_watts
p_ru_unused[0] = 100 - p_ru_used[0]
p_ru_unused[1] = 100 - p_ru_used[1]
p_ru_unused[2] = 100 - p_ru_used[2]

np.savetxt('num_RBs_em.csv', num_RBs_em, delimiter=',')
np.savetxt('num_RBs_ur.csv', num_RBs_ur, delimiter=',')
np.savetxt('num_RBs_mm.csv', num_RBs_mm, delimiter=',')
np.savetxt('num_RBs_unused.csv', num_RBs_unused, delimiter=',')
np.savetxt('p_ru_used.csv', p_ru_used, delimiter=',')
np.savetxt('p_ru_unused.csv', p_ru_unused, delimiter=',')
np.savetxt('p_ru_sca.csv', p_ru_sca, delimiter=',')
#np.savetxt('C:\\Users\\Admin-G212\\Desktop\\drl_v7\\value_bb.csv', value_bb, delimiter=',')
#np.savetxt('C:\\Users\\Admin-G212\\Desktop\\drl_v7\\value_sca.csv', value_sca, delimiter=',')
#np.savetxt('C:\\Users\\Admin-G212\\Desktop\\drl_v7\\value_ben.csv', value_ben, delimiter=',')
#p.savetxt('C:\\Users\\Admin-G212\\Desktop\\drl_v7\\R_em_sca.csv', R_sca, delimiter=',')
#np.savetxt('C:\\Users\\Admin-G212\\Desktop\\drl_v7\\L_ur_sca.csv', L_sca, delimiter=',')
#np.savetxt('C:\\Users\\Admin-G212\\Desktop\\drl_v7\\R_em_bb.csv', R_bb, delimiter=',')
#np.savetxt('C:\\Users\\Admin-G212\\Desktop\\drl_v7\\L_ur_bb.csv', L_bb, delimiter=',')
#np.savetxt('C:\\Users\\Admin-G212\\Desktop\\drl_v7\\R_em_ben.csv', R_ben, delimiter=',')
#np.savetxt('C:\\Users\\Admin-G212\\Desktop\\drl_v7\\L_ur_ben.csv', L_ben, delimiter=',')

'''
#################### Plot graph ########################
plt.rcParams['font.sans-serif'] = "Cambria"
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = "sans-serif"
# Sample data for the bar graph
points = []
for k in range(U_em):
    points.append(f'User {k}')


# Bar width
bar_width = 0.3
# Create a figure with two subplots
#plt.figure(figsize=(15, 15))
fig, axs = plt.subplots(figsize=(8, 4))
#plt.subplot(2, 2, 1)
axs.plot(value_bb[1:],'*-',label='BB', color="#EA4335")
axs.plot(value_sca[1:],'o-',label='SCA', color="#4285F4")
axs.plot(value_ben[1:],'s-',label='Benchmark', color ="#FBBC04")
axs.set_xlabel('iteration')
axs.set_ylabel('Objective Value')
axs.legend()
current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
fig_name = fr"pic/Pic_{datetime_string}.png"
plt.savefig(fig_name)
plt.show(block=False)
plt.pause(3)
plt.close()



fig, axs = plt.subplots(1,2,figsize=(11, 7))
# Stack the bars
x = ['BB', 'SCA', 'Benchmark']

bars1 = axs[0].bar(x, num_RBs_em, label='EMBB', color="#EA4335",edgecolor='black', linewidth=1)
bars2 = axs[0].bar(x, num_RBs_mm, bottom=num_RBs_em, label='MMTC',color="#4285F4",edgecolor='black', linewidth=1)
bars3 = axs[0].bar(x, num_RBs_ur, bottom=[sum(x) for x in zip(num_RBs_em, num_RBs_mm)], label='URLLC',color ="#FBBC04",edgecolor='black', linewidth=1)
bars4 = axs[0].bar(x, num_RBs_unused, bottom=[sum(x) for x in zip(num_RBs_em, num_RBs_mm, num_RBs_ur)], label='unused',color = "#FFFFFF",edgecolor='black', linewidth=1)

# Add labels and title
axs[0].set_xlabel('(a)', fontsize=13)
axs[0].set_ylabel('RB Usage (%)', fontsize=13)
axs[0].tick_params(axis='y', labelsize=10)
axs[0].set_ylim([0,102])
#ax.set_title('RB Usage per RRH')

# Add legend
axs[0].legend(loc='upper left',fontsize=10)

axs[1].bar(x, p_ru_used, label='Used Power', color = "#34A853",edgecolor='black', linewidth=1)
axs[1].bar(x, p_ru_unused, bottom=p_ru_used, label='Unused Power', color = "#FFFFFF",edgecolor='black', linewidth=1)
#axs[0,0].bar(x, num_RBs_ur, bottom=[sum(x) for x in zip(num_RBs_em, num_RBs_mm)], label='URLLC')
#axs[0,0].bar(x, num_RBs_unused, bottom=[sum(x) for x in zip(num_RBs_em, num_RBs_mm, num_RBs_ur)], label='unused')
axs[1].set_ylabel('Power Usage (%)', fontsize=13)
axs[1].set_xlabel('(b)', fontsize=13)
#axs[0].set_xticks([]) #hide x tick
axs[1].tick_params(axis='y', labelsize=10)
axs[1].set_ylim([0,102])
axs[1].legend(loc='upper left',fontsize=10)

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
fig_name = fr"pic/Pic_{datetime_string}.png"
plt.savefig(fig_name)
plt.show(block=False)
plt.pause(3)
plt.close()
'''
#plt.subplot(2, 2, 3)
#bar_positions = np.arange(len(points))
#axs[1, 0].bar(bar_positions - bar_width, R_bb, width=bar_width, label='R_bb')
#axs[1, 0].bar(bar_positions, R_sca, width=bar_width, label='R_sca')
#axs[1, 0].bar(bar_positions + bar_width, R_ben, width=bar_width, label='R_Ben')
#axs[1, 0].set_xlabel('User')
#axs[1, 0].set_ylabel('Throughput')
#axs[1, 0].set_xticks(bar_positions, points)
#axs[1, 0].legend()

#point1s = []
#for k in range(U_ur):
#    point1s.append(f'User {k}')
#plt.subplot(2, 2, 4)
#bar_positions = np.arange(len(point1s))
#plt.bar(bar_positions - bar_width, L_bb, width=bar_width, label='L_bb')
#plt.bar(bar_positions, L_sca, width=bar_width, label='L_sca')
#plt.bar(bar_positions + bar_width, L_ben, width=bar_width, label='L_Benc')
#plt.xlabel('User')
#plt.ylabel('Latency')
#plt.xticks(bar_positions, point1s)
#plt.legend()

#point1s = []
#for k in range(num_RUs):
#    point1s.append(f'RU {k}')
#plt.subplot(2, 2, 4)
#bar_positions = np.arange(len(point1s))
#axs[0, 0].bar(bar_positions - bar_width, p_ru_bb, width=bar_width, label='L_bb')
#axs[0, 0].bar(bar_positions, p_ru_sca, width=bar_width, label='L_sca')
#axs[0, 0].bar(bar_positions + bar_width, p_ru_ben, width=bar_width, label='L_Benc')
#axs[0, 0].set_xlabel('RU')
#axs[1, 1].set_ylabel('Total power')
#axs[1, 1].set_xticks(bar_positions, point1s)
#axs[1, 1].legend()

#current_datetime = datetime.datetime.now()
#datetime_string = current_datetime.strftime("%Y%m%d%H%M%S")
#fig_name = fr"pic/Pic_{datetime_string}.png"
#plt.savefig(fig_name)
#plt.show(block=False)
#plt.pause(3)
#plt.close()
