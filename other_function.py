import numpy as np
import pickle as pkl
import gzip as gz
import datetime
import time
from fnmatch import fnmatch
import shutil
import os

def extract_values(array, dtype):
        shape = array.shape
        flat_array = np.array([x.value for x in array.flatten()], dtype=dtype)
        return flat_array.reshape(shape)


def extract_optimization_results(pi_sk, z_ib_sk, p_ib_sk, mu_ib_sk, phi_i_sk, phi_j_sk, phi_m_sk):
    
    arr_pi_sk = extract_values(pi_sk, int)
    arr_z_ib_sk = extract_values(z_ib_sk, int)
    arr_p_ib_sk = extract_values(p_ib_sk, float)
    arr_mu_ib_sk = extract_values(mu_ib_sk, float)
    arr_phi_i_sk = extract_values(phi_i_sk, int)
    arr_phi_j_sk = extract_values(phi_j_sk, int)
    arr_phi_m_sk = extract_values(phi_m_sk, int)

    return arr_pi_sk, arr_z_ib_sk, arr_p_ib_sk, arr_mu_ib_sk, arr_phi_i_sk, arr_phi_j_sk, arr_phi_m_sk


def generate_new_num_UEs(num_UEs, delta_num_UE):
    # Tính sai số ngẫu nhiên trong khoảng [-delta_num_UE, delta_num_UE]
    delta = np.random.randint(-delta_num_UE, delta_num_UE)

    # Tính số lượng UE mới
    new_num_UEs = num_UEs + delta
    # Đảm bảo số lượng UE không âm
    return max(new_num_UEs, 0)

def mapping_RU_UE(slice_mapping, distances_RU_UE):
    num_RU, num_UEs = distances_RU_UE.shape
    num_slices, _ = slice_mapping.shape
    
    # Khởi tạo ma trận nearest_phi_i_sk
    nearest_phi_i_sk = np.zeros((num_RU, num_slices, num_UEs), dtype=int)
    
    for j in range(num_UEs):  # Duyệt qua từng UE
        for k in range(num_slices):  # Duyệt qua từng slice
            if slice_mapping[k, j] == 1:  # Kiểm tra UE thuộc slice nào
                # Tìm RU gần nhất với UE j
                nearest_RU = np.argmin(distances_RU_UE[:, j])
                # Đánh dấu ánh xạ trong nearest_phi_i_sk
                nearest_phi_i_sk[nearest_RU, k, j] = 1
    
    return nearest_phi_i_sk


def save_object(filename, object):
    with gz.open(filename, mode="wb", compresslevel=9) as f:
        f.write(
            pkl.dumps(object)
        )
        
def load_object(filename):
    with gz.open(filename, mode='rb') as f:
        return pkl.load(f)
    
class Stopwatch:
    def __init__(self, name, silent=True):
        self.name = name
        self.createat = datetime.datetime.now()
        self.startedat = None
        self.stopped = False
        self.checkpoints = []
        self.i = 0
        self.silent = silent
        
    def start(self):
        if self.stopped:
            raise Exception("Cannot start a stopped instance !!!")
        self.startedat = datetime.datetime.now()
        timestamp = round(time.process_time(), 5)
        r = (timestamp, "Started", 0)
        self.checkpoints.append(
            r
        )
        if not self.silent:
            print(f"StopWatch={self.name} | {self.__row2str(r)}")
        self.i += 1
    
    def add(self, contents=None):
        if self.stopped:
            raise Exception("Cannot add checkpoing to a stopped instance !!!")
        if self.startedat is None:
            raise Exception("Cannot add checkpoing to a not-yet-started instance !!!")
        if contents is None:
            contents = f"Checkpoint_{self.i}"
        timestamp = round(time.process_time(), 5)
        diff = round(timestamp - self.checkpoints[self.i-1][0], 5)
        r = (timestamp, contents, diff)
        self.checkpoints.append(
            r
        )
        self.i += 1
        if not self.silent:
            print(f"StopWatch={self.name} | {self.__row2str(r)}")
        
    def stop(self):
        if self.stopped:
            return
        if self.startedat is None:
            raise Exception("Cannot stop a not-yet-started instance !!!")
        self.stopped = True
        timestamp = round(time.process_time(), 5)
        diff = round(timestamp - self.checkpoints[self.i-1][0], 5)
        r = (timestamp, "Stopped", diff)
        self.checkpoints.append(
            r
        )
        self.i += 1
        if not self.silent:
            print(f"StopWatch={self.name} | {self.__row2str(r)}")

    def __repr__(self):
        return f"stopwatch name={self.name} created={self.createat.strftime('%Y%m%d_%H%M%S')} started={not (self.startedat is None)} stopped={self.stopped} n={self.i}"
    
    def __row2str(self, r):
        return f"{r[0]},{r[2]},{r[1]}"
        
    def write_to_file(self, filename):
        with open(filename, "wt") as f:
            f.write("time,diff,contents\n")
            for r in self.checkpoints:
                f.write(
                    self.__row2str(r)+"\n"
                )
                
def RecurseListDir(root: str, pattern: list[str]):
    f = []
    for p in pattern:
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, p):
                    f.append(os.path.join(path, name))
    return f