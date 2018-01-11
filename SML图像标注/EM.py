import numpy as np

'''
EM.py文件定义了第一个EM算法估计GMM的过程
'''

def gauss_process(x, miu, sigma):
    '''定义高斯分布'''
    gaussin=(1.0/(np.power(2*np.pi,0.5)*sigma))*np.exp(((-1.0)*np.power((x-miu),2))/(2*np.power(sigma,2)))
    return gaussin

def L_func(data_array, mean_array, cov_array, weight_array, N, K):
    '''
    定义似然函数
    x_array:(N,1)
    miu_array:(1,K)
    sigma_array:(1,K)
    pi_array:(1,K)
    '''
    loss=[]
    data_array=np.reshape(data_array, (1, N))
    assert (np.shape(data_array) == (1, N))
    for k in range(K):
        loss.append(weight_array[0][k] * gauss_process(data_array[0], mean_array[0][k], cov_array[0][k]))
    assert (np.shape(loss)==(K,N))
    loss=np.sum(loss,axis=0)
    loss=np.log(loss)
    loss=np.reshape(loss,(1,N))
    loss=np.sum(loss)
    return loss

def E_step(data_array, mean_array, cov_array, weight_array, N, K):
    '''
    E-step，其中N区域个数,k为每个GMM的高斯分布个数
    x_array:(N,1)
    miu_array:(1,K)
    sigma_array:(1,K)
    pi_array:(1,K)
    '''
    gaussin_sum=[]
    gamma=[]
    data_array=np.reshape(data_array, (1, N))
    for k in range(K):
        gaussin_sum.append(weight_array[0][k] * gauss_process(data_array[0, :], mean_array[0][k], cov_array[0][k]))
    gaussin_sum=np.array(gaussin_sum)
    gaussin_sum=np.reshape(gaussin_sum,(K,N))
    gaussin_sum=np.sum(gaussin_sum,axis=0)
    gaussin_sum=np.reshape(gaussin_sum,(1,N))
    assert (np.shape(gaussin_sum)==(1,N))
    for k in range(K):
        gamma.append((weight_array[0][k] * gauss_process(data_array[0, :], mean_array[0][k], cov_array[0][k])) / gaussin_sum)
    gamma=np.array(gamma,dtype=float)
    gamma=np.reshape(gamma,(K,N))
    assert(gamma.shape==(K,N))
    return gamma

def M_step(data_array, N, K, gamma_array):
    '''
    M-step，其中N区域个数,k为每个GMM的高斯分布个数
    x_array:(N,1)
    miu_array:(1,K)
    sigma_array:(1,K)
    pi_array:(1,K)
    gamma_array:(K,N)
    '''
    Nk=np.sum(gamma_array,axis=1)#(K,1)
    Nk=np.reshape(Nk,(K,1))
    assert (np.shape(Nk)==(K,1))
    pik=Nk/N#(K,1)
    pik=np.reshape(pik,(1,K))
    miu=[]
    sigma=[]
    data_array=np.reshape(data_array, (1, N))#(1,N)
    for k in range(K):
        miu.append((1.0/Nk[k][0]) * np.sum(gamma_array[k,:] * data_array[0, :]))
    miu=np.array(miu)
    miu=np.reshape(miu,(1,K))
    assert (np.shape(miu)==(1,K))
    for k in range(K):
        sigma.append((1.0/Nk[k][0]) * np.sum(np.dot(gamma_array[k,:] * (data_array - miu[0][k]), np.transpose(data_array - miu[0][k]))))
    sigma=np.array(sigma)
    sigma=np.reshape(sigma,(1,K))
    assert (np.shape(sigma)==(1,K))
    return pik,miu,sigma

def process(data,iter_count,N,K,threshold,D):
    '''
    第一个EM算法过程
    data:(D,N,30)
    '''
    final_miu=[]
    final_sigma=[]
    final_pi=[]
    for picture in range(D):
        temp_miu = []
        temp_sigma = []
        temp_pi = []
        x_array=data[picture,:,:]
        x_array=np.array(x_array,dtype=float)
        x_array=np.reshape(x_array,(N,30))
        # print('p_count',picture)
        for i in range(30):
            # print('i_count',i)
            temp_x_array=x_array[:,i]
            temp_x_array=np.reshape(temp_x_array,(N,1))
            miu_array=[]
            for i in range(8):
                miu_array.append(9)
            miu_array=np.array(miu_array,dtype=float)
            miu_array=np.reshape(miu_array, (1, K))
            # sigma_init = ()
            # for n in range(K):
            #     sigma_init+=(np.random.choice(100))
            sigma_array=np.arange(1,9,1,dtype=float)
            sigma_array=np.reshape(sigma_array,(1,K))
            assert (np.shape(sigma_array)==(1,K))
            pi_array=np.array([[1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8,1.0/8]])
            for iter in range(iter_count):
                # print('iter_count',iter)
                gamma_array=E_step(temp_x_array,miu_array,sigma_array,pi_array,N,K)
                pi_array,miu_array,sigma_array=M_step(temp_x_array,N,K,gamma_array)
                loss=L_func(temp_x_array, miu_array, sigma_array, pi_array, N, K)
                # print(gamma_array)
                # print('temp_x_array',temp_x_array)
                # print('pi_array',pi_array)
                # print('miu_array',miu_array)
                # print('sigma_array',sigma_array)
                # print(loss)
                # if loss<threshold:
                #     break
            temp_miu.append(miu_array)
            temp_sigma.append(sigma_array)
            temp_pi.append(pi_array)
        temp_miu=np.array(temp_miu)
        temp_miu=np.reshape(temp_miu,(30,K))
        temp_sigma=np.array(temp_sigma)
        temp_sigma=np.reshape(temp_sigma,(30,K))
        temp_pi=np.array(temp_pi)
        temp_pi=np.reshape(temp_pi,(30,K))
        final_miu.append(temp_miu)
        final_sigma.append(temp_sigma)
        final_pi.append(temp_pi)
    final_miu=np.array(final_miu)
    final_miu=np.reshape(final_miu,(D,30,K))
    final_sigma=np.array(final_sigma)
    final_sigma=np.reshape(final_sigma,(D,30,K))
    final_pi=np.array(final_pi)
    final_pi=np.reshape(final_pi,(D,30,K))
    return final_miu,final_sigma,final_pi


