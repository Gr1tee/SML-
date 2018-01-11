import numpy as np

'''
expand_EM.py文件定义了第二个扩展EM算法用于GMM分量聚类的过程
'''

def gauss_process(x, miu, sigma):
    '''定义高斯分布'''
    gaussin=1.0/(np.power(2*np.pi,0.5)*sigma)*np.exp((-1)*np.power((x-miu),2)/(2*np.power(sigma,2)))
    return gaussin

def expand_E_step(meanj_array, covj_array, weightj_array, meanc_array, covc_array, weightc_array, M, D, K):
    '''
    E-step,其中M为新的高斯分布个数,K为原高斯分布个数，D为样本图片数
    miuj_array:(D,K)
    sigmaj_array:(D,K)
    pic_array:(D,K)
    miuc_array:(1,M)
    sigmac_array:(1,M)
    pic_array:(1,M)
    '''
    h_deno=[]
    h=[]
    for m in range(M):
        h_nume=[]
        for j in range(D):
            miu_array_j= meanj_array[j, :]
            miu_array_j=np.reshape(miu_array_j,(1,K))
            sigma_array_j= covj_array[j, :]
            sigma_array_j=np.reshape(sigma_array_j,(1,K))
            pi_array_j= weightj_array[j, :]
            pi_array_j=np.reshape(pi_array_j,(1,K))
            h_nume.append(np.power((gauss_process(miu_array_j, meanc_array[0][m], covc_array[0][m]) * np.exp((-0.5) * np.trace(sigma_array_j / covc_array[0][m]))), pi_array_j) * weightc_array[0][m])
        # h_nume=np.array(h_nume)
        # h_nume=np.reshape(h_nume,(D,K))
        h_deno.append(h_nume)
    h_deno=np.array(h_deno)
    h_deno=np.reshape(h_deno,(M,D,K))
    assert (np.shape(h_deno)==(M,D,K))
    h_deno=np.sum(h_deno,axis=0)
    h_deno=np.reshape(h_deno,(D,K))
    assert (np.shape(h_deno)==(D,K))
    for m in range(M):
        h_nume=[]
        for j in range(D):
            miu_array_j= meanj_array[j, :]
            miu_array_j=np.reshape(miu_array_j,(1,K))
            sigma_array_j= covj_array[j, :]
            sigma_array_j=np.reshape(sigma_array_j,(1,K))
            pi_array_j= weightj_array[j, :]
            pi_array_j=np.reshape(pi_array_j,(1,K))
            h_nume.append(np.power((gauss_process(miu_array_j, meanc_array[0][m], covc_array[0][m]) * np.exp((-0.5) * np.trace(sigma_array_j / covc_array[0][m]))), pi_array_j) * weightc_array[0][m])
        h_nume=np.array(h_nume)
        h_nume=np.reshape(h_nume,(D,K))
        h.append(h_nume/h_deno)
    h=np.reshape(h,(M,D,K))
    assert (np.shape(h)==(M,D,K))
    return h

def expand_M_step(h, meanj_array, covj_array, weightj_array, D, K, M):
    '''
    M-step,其中M为新的高斯分布个数,K为原高斯分布个数，D为样本图片数
    miuj_array:(D,K)
    sigmaj_array:(D,K)
    pij_array:(D,K)
    h:(M,D,K)
    '''
    pic_array=np.sum(h,axis=1)
    pic_array=np.reshape(pic_array,(M,K))
    pic_array=np.sum(pic_array,axis=1)
    pic_array=np.reshape(pic_array,(1,M))
    pic_array=pic_array/(D*K)
    assert (np.shape(pic_array)==(1,M))
    W=[]
    miuc=[]
    sigmac=[]
    for m in range(M):
        h_array=h[m,:,:]
        h_array=np.reshape(h_array,(D,K))
        w_deno=np.sum(h_array * weightj_array)
        w= h_array * weightj_array / w_deno
        w=np.reshape(w,(D,K))
        W.append(w)
    W=np.array(W)
    W=np.reshape(W,(M,D,K))
    for m in range(M):
        w_array=W[m,:,:]
        w_array=np.reshape(w_array,(D,K))
        miuc.append(np.sum(w_array * meanj_array))
    miuc=np.array(miuc)
    miuc=np.reshape(miuc,(1,M))
    for m in range(M):
        w_array=W[m,:,:]
        w_array=np.reshape(w_array,(D,K))
        temp_sigmac=[]
        for j in range(D):
            temp_sigmac.append(w_array[j,:] * (covj_array[j, :] + np.dot((meanj_array[j, :] - miuc[0][m]), np.transpose(meanj_array[j, :] - miuc[0][m]))))
        temp_sigmac=np.array(temp_sigmac)
        temp_sigmac=np.reshape(temp_sigmac,(D,K))
        # miuc_array=miuc[m,:,:]
        # miuc_array=np.reshape(miuc_array,(D,K))
        # np.sum(w_array * (sigmaj_array + (miuj_array - miuc[0][m]) * np.transpose(miuj_array - miuc[0][m])))
        sigmac.append(temp_sigmac)
    sigmac=np.array(sigmac)
    sigmac=np.reshape(sigmac,(M,D,K))
    sigmac=np.sum(sigmac,axis=1)
    sigmac=np.reshape(sigmac,(M,K))
    sigmac=np.sum(sigmac,axis=1)
    sigmac=np.reshape(sigmac,(1,M))
    return miuc,sigmac,pic_array

def expand_process(mean_array_j, cov_array_j, weight_array_j, K, M, D, iter_count):
    '''
    第二个EM算法过程
    miu_array_j:(D,30,K)
    sigma_array_j:(D,30,K)
    pi_array_j:(D,30,K)
    output:
    final_miu:(30,M)
    final_sigma:(30,30,M)
    final_pi:(30.M)
    '''
    final_miu=[]
    final_sigma=[]
    final_pi=[]
    for i in range(30):
        print('ex_i_count',i)
        miuj_array= mean_array_j[:, i, :]
        miuj_array=np.reshape(miuj_array,(D,K))
        sigmaj_array= cov_array_j[:, i, :]
        sigmaj_array=np.reshape(sigmaj_array,(D,K))
        pij_array= weight_array_j[:, i, :]
        pij_array=np.reshape(pij_array,(D,K))
        miuc_array=np.random.randn(1,M)+1
        sigmac_array=np.random.randn(1,M)+1
        pic_array=np.random.randn(1,M)+1
        for iter in range(iter_count):
            print('ex_iter_count',iter)
            h=expand_E_step(miuj_array,sigmaj_array,pij_array,miuc_array,sigmac_array,pic_array,M,D,K)
            miuc_array,sigmac_array,pic_array=expand_M_step(h,miuj_array,sigmaj_array,pij_array,D,K,M)
        final_miu.append(miuc_array)
        final_sigma.append(sigmac_array)
        final_pi.append(pic_array)
    final_miu=np.array(final_miu)
    final_miu=np.reshape(final_miu,(30,M))
    final_sigma=np.array(final_sigma)
    final_sigma=np.reshape(final_sigma,(30,M))
    final_pi=np.array(final_pi)
    final_pi=np.reshape(final_pi,(30,M))
    return final_miu,final_sigma,final_pi










