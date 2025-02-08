import numpy as np
import subprocess
import mmap
import posix_ipc

SHM_NAME = "/shm_env"
SEM_READY = "/sem_ready"
SEM_DONE = "/sem_done"

OBS_ROWS, OBS_COLS = 500, 140
ACT_SIZE = 10000
OBS_SIZE = OBS_ROWS * OBS_COLS
MEM_SIZE = (ACT_SIZE + OBS_SIZE + 1) * 4  # 4 bytes per float

class Environment:
    def __init__(self):
        self.cpp_executable = "./build/lifelong"
        self.input_file = "./example_problems/random.domain/random_32_32_20_200.json"
        self.output_file = "./outputs/tmp.json"
        self.timesteps = "20"

        self.process = None

        # self.shm = posix_ipc.SharedMemory(SHM_NAME, posix_ipc.O_CREAT, size=MEM_SIZE)
        # self.map = mmap.mmap(self.shm.fd, MEM_SIZE)
        # self.sem_ready = posix_ipc.Semaphore(SEM_READY, posix_ipc.O_CREAT, initial_value=0)
        # self.sem_done = posix_ipc.Semaphore(SEM_DONE, posix_ipc.O_CREAT, initial_value=0)

    def reset(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

        cmd = [
            self.cpp_executable,
            "--inputFile", self.input_file,
            "-o", self.output_file,
            "-s", self.timesteps
        ]

        self.process = subprocess.Popen(cmd)


    # def step(self, actions):
    #     """ Send step signal and actions, then wait for observations """
    #     # Write step signal and actions
    #     data = np.zeros(ACT_SIZE + OBS_SIZE + 1, dtype=np.float32)
    #     data[0] = 1  # Step signal
    #     data[1:ACT_SIZE+1] = actions
    #     self.map.seek(0)
    #     self.map.write(data.tobytes())

    #     # Notify C++ that data is ready
    #     self.sem_ready.release()
    #     self.sem_done.acquire()  # Wait for response

    #     # Read observation matrix
    #     self.map.seek((ACT_SIZE + 1) * 4)  # Skip step signal & actions
    #     obs_matrix = np.frombuffer(self.map.read(OBS_SIZE * 4), dtype=np.float32).reshape(OBS_ROWS, OBS_COLS)
    #     return obs_matrix

    def close(self):
    #     """ Send stop signal and cleanup """
    #     self.map.seek(0)
    #     self.map.write(np.zeros(ACT_SIZE + OBS_SIZE + 1, dtype=np.float32).tobytes())
    #     self.sem_ready.release()
    #     self.map.close()
    #     self.shm.unlink()
    #     self.sem_ready.close()
    #     self.sem_done.close()

        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None








if __name__ == "__main__":
    env = Environment()
    env.reset()
    env.process.wait()
    env.close()
    
    

