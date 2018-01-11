import scipy.io as sio
import numpy as np

result=sio.loadmat('result.mat')
# print(result.keys())
pi=result['weight_ex']
miu=result['mean_ex']
sigma=result['covar_ex']
pi_array=pi
miu=np.reshape(miu,(30,64))
for i in range(2,31):
    pi_temp=np.hstack((pi[:,64-i:],pi[:,:64-i]))
    pi_array=np.vstack((pi_array,pi_temp))
print(pi_array)
print(np.shape(pi_array))
pi=pi_array
result={'mean':miu,'cov':sigma,'weight':pi}
sio.savemat('result1.mat',{'mean':result['mean'],'cov':result['cov'],'weight':result['weight']})



# import random
# def generate_rand(n, sum_v):
#     temp = random.randint(0, sum_v//20)
#     print(temp)
#     if n>0:
#         generate_rand(n-1, sum_v-temp)
#
# generate_rand(64,1)

