import device_run as device_run
import math
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

model = device_run.initiate()
gates = ["c3", "c4", "c5", "c6", "c7", "c9", "c10"]
gate_numbers = [3, 4, 5, 6, 7, 9, 10]
initial_gate_cu_cv_num_indices = (2, 5)
list_of_gates = [-1097.19870146, -1972.54652661, -1025.76251066,  -964.9129052,
 -1595.64035536, -1037.70755352,  -112.49633387]
epsilon = 0.0
pygor_xmlip = 'http://129.67.86.107:8000/RPC2'

print('go')
t0 = datetime.now()

env, run_information = device_run.run(model, epsilon, gates, gate_numbers, list_of_gates,
                                      initial_gate_cu_cv_num_indices, pygor_xmlip=pygor_xmlip, show_log=False,
                                      MaxStep=2, starting_position=[0,0], random_pixel=True,
                                      savefilename=None)

centre_voltages = env.block_centre_voltages[math.floor(env.allowed_n_blocks / 2.0)][
    math.floor(env.allowed_n_blocks / 2.0)]

data = env.pygor.do2d(env.control_gates[0], centre_voltages[0] + (env.allowed_n_blocks / 2.0) * env.block_size,
                      centre_voltages[0] - (env.allowed_n_blocks / 2.0) * env.block_size, env.allowed_n_blocks * env.block_size , env.control_gates[1],
                      centre_voltages[1] + (env.allowed_n_blocks / 2.0) * env.block_size,
                      centre_voltages[1] - (env.allowed_n_blocks / 2.0) * env.block_size, env.allowed_n_blocks * env.block_size )

#data = data.data

time_taken = datetime.now() - t0

scan_information = {"Scan data.data": data,
                    "Scan time (s)": time_taken.total_seconds()}

print(time_taken.total_seconds(),' Seconds')
pickle_out = open('benchmark/regime_4_full_scan_data_time'+".pickle","wb")
pickle.dump(scan_information, pickle_out)
pickle_out.close()

plt.imshow(data.data[0], extent=[env.block_centre_voltages[0][0][0], env.block_centre_voltages[-1][-1][0],
                         env.block_centre_voltages[-1][-1][1], env.block_centre_voltages[0][0][1]])
plt.colorbar()
plt.title("Full scan")
plt.show()

device_run.end_session(model)