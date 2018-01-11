import numpy as np
import scipy.io as sio
from EM import process
from expand_EM import expand_process


def gauss_process(x, miu, sigma):
    '''定义高斯分布'''
    gaussin=1.0/(np.power(2*np.pi,0.5)*sigma)*np.exp((-1)*np.power((x-miu),2)/(2*np.power(sigma,2)))
    return gaussin

def loadData(mat_path, D, N):
    data=sio.loadmat(mat_path)
    data=data['resultofdct']
    data=np.array(data,dtype=float)
    data=np.reshape(data,(D,N,30))
    return data

def main(path, iter_count, N, K, D, M, threshold):
    '''主函数'''
    '''读取DCT变换后的数据'''
    print('读取数据')
    data=loadData(path, D, N)
    '''
    训练过程
    miu,sigma,pi为训练得出的语义类w的条件分布64个Gauss的GMM参数
    '''
    print('EM1')
    miu_array_j,sigma_array_j,pi_array_j=process(data,iter_count,N,K,threshold,D)
    print('EM2')
    miu,sigma,pi=expand_process(miu_array_j, sigma_array_j, pi_array_j, K, M, D, iter_count)
    result={'miu':miu,'sigma':sigma,'pi':pi}
    '''
    保存结果
    '''
    sio.savemat('result.mat',{'miu':result['miu'],'sigma':result['sigma'],'pi':result['pi']})
    '''
    计算过程
    pxw为包含64个Gaussin的语义类w对某个图的每N个区域的30个DCT变换点的条件分布
    '''
    # pxw=[]
    # for m in M:
    #     temp_miu=miu[:,m]
    #     temp_miu=np.reshape(temp_miu,(1,30))
    #     temp_sigma=sigma[:,m]
    #     temp_sigma=np.reshape(temp_sigma,(1,30))
    #     temp_pi=pi[:,m]
    #     temp_pi=np.reshape(temp_pi,(1,30))
    #     pxw[m]=np.sum(temp_pi*gaussin_func(test_data,temp_miu,temp_sigma))
    # pxw=np.array(pxw)
    # pxw=np.reshape(pxw,(M,N,30))
    # pxw=np.sum(pxw,axis=0)
    # pxw=np.reshape(pxw,(N,30))
    # print(pxw)

if __name__ == '__main__':
    main('resultofdct.mat',1000,100,8,100,64,0.01)
