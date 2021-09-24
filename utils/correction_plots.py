import numpy as np
import matplotlib.pyplot as plt

np_file = np.load('losses.npy',allow_pickle=True)
np_file = np.array(np_file)
print(np_file)

np_file = np.load('fid_log.npy')
np_file = np.array(np_file)
print(np_file)

x_arr = np_file[0,:]
x_arr = np.delete(x_arr, [20,21,22,48])

print(x_arr)

y_arr =np_file[1,:]
y_arr = np.delete(y_arr, [20,21,22,48])

plt.figure()
plt.plot(x_arr, y_arr)
plt.xlabel('Iterations')
plt.ylabel('FID-Score')
plt.grid(b=True, which='major', color='#666666', linestyle='--')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
plt.savefig('FID_updated.eps', format = 'eps', dpi=600)
plt.close()

