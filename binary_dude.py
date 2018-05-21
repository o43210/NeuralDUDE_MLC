
# coding: utf-8

# In[1]:

import numpy as np
from numpy import *

# In[3]:

def bit_xor(a,b):
    return int(bool(a)^bool(b))

def error_rate(a,b):
    error=np.zeros(len(a))
    for i in range(len(a)):
        error[i]=bit_xor(a[i],b[i])
    return np.sum(error)/len(a)
    

def bsmc(n,alpha):
    a=np.zeros(n,dtype=np.int)
    a[0]=int(np.random.random()>0.5)

    for i in range(int(n)-1):
        trans=int(np.random.random()<alpha)
        a[i+1]=bit_xor(a[i],trans) 
    return a

def bsc(x,delta):
    z=np.zeros(len(x),dtype=np.int)
    for i in range(len(x)):
        noise=int(np.random.random()<delta)
        z[i]=bit_xor(x[i],noise)
    return z

def dude(z,k,delta):
   # print "Running DUDE algorithm"
    n=len(z)
    x_hat=np.zeros(n,dtype=np.int)

    th_0=2*delta*(1-delta)
    m={}
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if not m.has_key(context_str):
            m[context_str]=np.zeros(2,dtype=np.int)
            m[context_str][z[i]]=1
        else:
            m[context_str][z[i]]+=1
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)

        ratio = float(m[context_str][z[i]]) / float(np.sum(m[context_str]))
        if ratio >= th_0:
            x_hat[i]=z[i]
        else:
            x_hat[i]=int(not bool(z[i]))

    return x_hat

def dude2(z,k,delta):
   # print "Running DUDE algorithm"
    n=len(z)
    x_hat=np.zeros(n,dtype=np.int)
    s_hat=x_hat.copy()

    th_0=2*delta*(1-delta)
    th_1=delta**2+(1-delta)**2
    
    m={}
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if not m.has_key(context_str):
            m[context_str]=np.zeros(2,dtype=np.int)
            m[context_str][z[i]]=1
        else:
            m[context_str][z[i]]+=1
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        ratio = float(m[context_str][1]) / float(np.sum(m[context_str]))
        
        if ratio < th_0:
            s_hat[i]=1
        elif ratio >= th_1:
            s_hat[i]=2
        else:
            s_hat[i]=0

    return s_hat, m
     
def denoise_with_s(z,s,k):
    n=len(z)
    x_hat=z.copy()
    for i in range(k,n-k):
        if s[i]==0:
            x_hat[i]=z[i]
        elif s[i]==1:
            x_hat[i]=0
        else:
            x_hat[i]=1
    return x_hat

def make_data_for_ndude(Z,k,L,nb_classes,n):
    c_length=2*k
    C=zeros((n-2*k, 2*k*nb_classes))

    for i in range(k,n-k):
        c_i = vstack((Z[i-k:i,],Z[i+1:i+k+1,])).reshape(1,2*k*nb_classes)
        C[i-k,]=c_i
        
    Y=dot(Z[k:n-k,],L)    
    return C,Y

def make_binary_image(im):
    im_bin=im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j]>127: 
                im_bin[i,j]=1
            else:
                im_bin[i,j]=0
    
    return im_bin


#############################CUDE###################################
def denoise_with_prob(z, k, delta, prob):
    n = len(z)
    x_hat = np.zeros(n)
    
    th=2*delta*(1-delta)
    
    for i in range(0,n-2*k):
        ratio = prob[i][z[i+k]]
        if ratio >= th:
            x_hat[i+k]=z[i+k]
        else:
            x_hat[i+k]=int(not bool(z[i+k]))

    return x_hat

def find_delta_cude(z, C, k, prob, context_cnt, delta, nb_classes):
    T = C.shape[0]
    n = len(z)
    gamma = zeros((T, nb_classes))
    P_ct = np.zeros(n - 2*k)
    for i in range(20):
        loop=0
        pi = np.array([[1 - delta, delta],[delta, 1-delta]])
        pi_inv = np.array([[1-delta, -delta], [-delta, 1-delta]]) / (1 - 2*delta)
        for t in range(T):
            if loop == 0:
                context = ''.join(str(e) for e in C[t].tolist())
                P_ct[t] = float(context_cnt[context]) / float(T)
            P_zt_ct = prob[t]
            P_ztct = prob[t] * P_ct[t]
            value = pi * (matmul(P_ztct,pi_inv).reshape(nb_classes, 1)) / P_ztct
            gamma[t] = value[:,z[t+k]]
        b = np.zeros((nb_classes, nb_classes))
        np.add.at(b.T, z[k+1:n-k+1], gamma)
        b /= (np.sum(gamma, axis = 0).reshape(nb_classes,1) + 1e-35)
        delta_before = delta
        delta = b[0][1]
        loop+=1
        if(abs(delta-delta_before) < 1e-6):
            break
    return delta

def make_data_for_cude(Z,k,n):
    c_length=2*k
    C=zeros((n-2*k, 2*k))
    context_cnt = {}
    
    for i in range(k,n-k):
        c_i = hstack((Z[i-k:i,1],Z[i+1:i+k+1,1]))
        c_i_str = ''.join(str(e) for e in c_i.tolist())
        if not context_cnt.has_key(c_i_str):
            context_cnt[c_i_str] = 1
        else:
            context_cnt[c_i_str] += 1
        C[i-k,]=c_i
        
    Y = Z[k:n-k]
    
    return C,Y,context_cnt

####################################################################

#############################qsc_HMM################################
def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def q_error_rate(x, z):
    n = len(x)
    error=np.zeros(n)
    for i in range(n):
        error[i]= x[i] != z[i]

    return np.sum(error)/n

def sym_mat(states, prob):
    x = ones((states,states)) * (prob/(states-1))
    for i in range(states):
        x[i][i] = 1 - (states-1)*x[i][i]
    return x

def qsmc(n, alpha):
    x = np.zeros(n, dtype = np.int)
    b = []
    for i in range(4):
        arr = [t for t in range(4) if t != i]
        b.append(arr)
        b[i].append(i)
    
    x[0] = int(np.random.random()/0.25)
    for i in range(1,int(n)):
        idx = min(int(np.random.random() / (alpha/3)),3)
        x[i] = b[x[i-1]][idx]
        
    return x        

def qsc(x, delta):
    n = len(x)
    z = np.zeros(n, dtype = np.int)
    b = []
    for i in range(4):
        arr = [t for t in range(4) if t != i]
        b.append(arr)
        b[i].append(i)
    
    for i in range(n):
        idx = min(int(np.random.random() / (delta/3)),3)
        z[i] = b[x[i]][idx]

    return z

def qbc(n, alpha, beta):
    x = np.zeros(n, dtype = np.int)
    x[0]=int(np.random.random()>0.5)
    x[1]=int(np.random.random()>0.5)
    pi = np.array([alpha, 1-alpha, beta, 1-beta])
    
    for i in range(2, n):
        state = x[i-2]*2 + x[i-1]
        x[i] = np.random.random() < pi[state]
        
    return x

def baum_welch(pi, a ,b ,x, denoise = True):
    T = x.shape[0]
    states = a.shape[0]
    obs_states = b.shape[1]
    gamma = None
    delta = None
    p = None
    while True:
        xi = np.zeros((T+1, 2, states))
        gamma = np.zeros((T+1, states))
        joint = np.zeros((T+1, states, states))
        
        for t in range(1,T+1): # 1~T
            eta = b[:, x[t-1]]
            if t==1:
                xi[t][0] = pi
            else:
                xi[t][0] = np.matmul(xi[t-1][1], a)
            xi[t][1] = (eta * xi[t][0]) / (np.sum(eta * xi[t][0]) + 1e-35)

        gamma[T] = xi[T][1]
        for t in reversed(range(1,T)):
            gamma[t] = xi[t][1] * np.matmul(gamma[t+1] / (xi[t+1][0]) + 1e-35, a.T)
            joint[t] = xi[t][1].reshape(states,1) * (gamma[t+1] / (xi[t+1][0] + 1e-35)) * a
            

        a_before = a
        b_before = b
        pi = gamma[1]
        a = np.sum(joint[1:T], axis = 0) / (np.sum(gamma[1:T], axis = 0).reshape(states,1) + 1e-35)
        b = np.zeros((states, obs_states))
        np.add.at(b.T, x, gamma[1:])
        b /= (np.sum(gamma, axis = 0).reshape(states,1) + 1e-35)
        
        if rel_error(a, a_before) < 1e-6 and rel_error(b, b_before) < 1e-6:
            break
    if denoise == False:
        return a, b, gamma
    
    #denoise the sequence
    x_hat = np.zeros(T)
    for t in range(T):
        x_hat[t] = np.argmax(gamma[t+1])
        
    return x_hat, a, b, gamma
        
def FB_recursion(pi, a, b, x, denoise = True):
    T = x.shape[0]
    states = a.shape[0]
    xi = np.zeros((T+1, 2, states))
    gamma = np.zeros((T+1, states))
    
    
    for t in range(1,T+1): # 1~T
        eta = b[:, x[t-1]]
        if t==1:
            xi[t][0] = pi
        else:
            xi[t][0] = np.matmul(xi[t-1][1], a)    
        xi[t][1] = (eta * xi[t][0]) / (np.sum(eta * xi[t][0]) + 1e-35)
        
        gamma[T] = xi[T][1]
    for t in reversed(range(1,T)):
        gamma[t] = xi[t][1] * np.matmul(gamma[t+1] / (xi[t+1][0] + 1e-35) , a.T)
    
    if denoise == False:
        return gamma
    #denoise the sequence
    x_hat = np.zeros(T)
    for t in range(T):
        x_hat[t] = np.argmax(gamma[t+1])

    return x_hat, gamma
####################################################################


# In[7]:

#n=10000
#print n
#alpha=0.1
#delta=0.1
#
#x=bsmc(n,alpha)
## print x
#
#z=bsc(x,delta)
## print z
#
#error = error_rate(x,z)
#print error
#
#err_k=np.zeros(6)
#for k in range(0,6):
#    x_hat= dude(z,k,delta)
#
#    error =error_rate(x,x_hat)
#    err_k[k]=error
#    print error
#    
#k=range(0,6)
#plt.plot(k,err_k)
# print m['1111']
# print m['1111'][0]
# print np.sum(m['1111'])
# print float(m['1111'][0]) / float(np.sum(m['1111']))
        
    



# In[ ]:



