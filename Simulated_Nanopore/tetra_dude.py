
# coding: utf-8

# In[1]:

from numpy import *
import datetime
    
# In[4]:
def transition(a,trans):
    if trans == 1:
        rand = random.random()
        if rand < 1.0/3.0:
            return (a+1)%4
        elif rand < 2.0/3.0:
            return (a+2)%4
        else:
            return (a+3)%4
    else:
        return a

def tsmc(n,alpha):
    a=zeros(n,dtype=int)
    rand = random.random()
    if rand < 0.25:
        a[0] = 0
    elif rand < 0.5:
        a[0] = 1
    elif rand < 0.75:
        a[0] = 2
    else:
        a[0] = 3

    for i in range(int(n)-1):
        trans=int(random.random()<alpha)
        a[i+1]=transition(a[i],trans) 
    return a


def PREPROCESS(lines,nt_order):
    z    = zeros(2500000,dtype=int)
    zn = 0
    for t in range(len(lines)):
        if t % 2 == 0:
            continue
     
        for i in range(len(lines[t])-1):
            if zn == len(z):
                break
            if nt_order.find(lines[t][i]) < 0:
                z[zn] = random.randint(0,4)
                zn += 1
                continue
            for j in range(4):
                if lines[t][i] == nt_order[j]:
                    z[zn] = j
                    zn += 1
                    break

    return z[:zn]

def POSTPROCESS(lines,nt_order,x_hat,f):
    zn = 0
    for t in range(len(lines)):
        if t % 2 == 0:
            f.write(lines[t])
            continue
     
        for i in range(len(lines[t])-1):
            f.write(nt_order[x_hat[zn]])
            zn += 1
        f.write('\n')
        
def error_rate(a,b):
    error=zeros(len(a))
    for i in range(len(a)):
        error[i]=int(a[i]!=b[i])
    return sum(error)/len(a)  

def make_data_for_ndude(z,Z,k,L,L_R,nb_classes,n):
    C=zeros((n-2*k, 2*k*nb_classes))
    CONT = zeros((n-2*k, 2*k*nb_classes))
    DIST = zeros((n-2*k, 4))
    ct = 0
    
    m={}    
        
    for i in range(k,n-k):
        c_i = vstack((Z[i-k:i,],Z[i+1:i+k+1,])).reshape(1,2*k*nb_classes)
        C[i-k,]=c_i        
        
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        if not m.has_key(context_str):
            m[context_str] = ct
            DIST[ct,z[i]] += 1
            CONT[ct,]= c_i
            ct += 1
        else:
            DIST[m[context_str],z[i]] += 1            
               
    
    Y=dot(Z[k:n-k,],L)                    #Original
    Y_R=dot(Z[k:n-k,],L_R)                #Reduced
    
    return C, Y, Y_R, CONT[:ct,], DIST[:ct,]

    
def dude(z,k,H,LAMBDA,PI):
   # print "Running DUDE algorithm"
    n=len(z)
    x_hat=z.copy()
    s_hat=z.copy()
    
    m={}
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if not m.has_key(context_str):
            m[context_str]=zeros(4,dtype=int)
            m[context_str][z[i]]=1
        else:
            m[context_str][z[i]]+=1

    s_context={}
    for context_str in m.keys():
        S = zeros(4,dtype=int)
        S_err=zeros(4,dtype=float)
        for q in range(4):
            for j in range(4):
                S_err[j] = dot(dot(m[context_str],H),LAMBDA[:,j]*PI[:,q])
            S[q] = argmin(S_err)
        s_context[context_str] = S[0] + S[1]*4 + S[2]*4*4 + S[3]*4*4*4
            
    for i in range(n):
        if i < k or n-k <= i:
            s_hat[i] = 228
            continue
            
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        s_hat[i] = s_context[context_str]

    return s_hat
     

def denoise_with_s(z,s,k):
    n=len(z)
    x_hat=z.copy()
    for i in range(k,n-k):
        s_index = s[i]
        DN = zeros(4)
        DN[3] = int(s_index / (4**3))
        s_index -= DN[3] * (4**3)
        DN[2] = int(s_index / (4**2))
        s_index -= DN[2] * (4**2)
        DN[1] = int(s_index / (4**1))
        s_index -= DN[1] * (4**1)
        DN[0] = s_index / (4**0)
           
        x_hat[i] = DN[z[i]]     
            
    return x_hat

def denoise_with_s_R(z,s,k):
    n=len(z)
    x_hat=z.copy()
    for i in range(k,n-k):  
        S = zeros(4)
        
        for j in range(4):
            S[j] = s[i][z[i]*4+j] 
            
        x_hat[i] = argmax(S)   
        
    return x_hat

def s_R_preprocess(s_pre, k):
    n = s_pre['output0'].shape[0] + 2*k
    s_nn_R_hat=zeros((n, 16))
    for i in range(n):
        if i<k or n-k<=i:
            s_i = array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
        else:
            s_i_temp = zeros(16,dtype=int)
            
            if max(s_pre['output0'][i-k]) == s_pre['output0'][i-k][0]:
                s_i_temp[0] = 1
            else:
                s_i_temp[argmax(s_pre['output0'][i-k])] = 1
                
            if max(s_pre['output1'][i-k]) == s_pre['output1'][i-k][1]:
                s_i_temp[5] = 1
            else:
                s_i_temp[argmax(s_pre['output1'][i-k])+4] = 1
                
            if max(s_pre['output2'][i-k]) == s_pre['output2'][i-k][2]:
                s_i_temp[10] = 1
            else:
                s_i_temp[argmax(s_pre['output2'][i-k])+8] = 1
                
            if max(s_pre['output3'][i-k]) == s_pre['output3'][i-k][3]:
                s_i_temp[15] = 1
            else:
                s_i_temp[argmax(s_pre['output3'][i-k])+12] = 1

            
            s_i = hstack(s_i_temp)
        s_nn_R_hat[i,]=s_i
    return s_nn_R_hat


def best_denoiser(x,z,k):
    n=len(z)
    x_hat=z.copy()
    s_hat=z.copy()
    
    m={}
    
    for i in range(k,n-k):
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        
        if not m.has_key(context_str):
            m[context_str]=zeros((4,4),dtype=int)
            m[context_str][z[i]][x[i]]=1
        else:
            m[context_str][z[i]][x[i]]+=1

    s_context={}
    for context_str in m.keys():
        S = zeros(4,dtype=int)
        for q in range(4):
            S[q] = argmax(m[context_str][q])
        s_context[context_str] = S[0] + S[1]*4 + S[2]*4*4 + S[3]*4*4*4
        
        
    for i in range(n):
        if i < k or n-k <= i:
            s_hat[i] = 228
            continue
            
        context=z[i-k:i].tolist()+z[i+1:i+k+1].tolist()
        context_str = ''.join(str(e) for e in context)
        s_hat[i] = s_context[context_str]

    return s_hat





def make_binary_image(im):
    im_bin=im.copy()
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j]>127: 
                im_bin[i,j]=1
            else:
                im_bin[i,j]=0
    
    return im_bin

def PRINT(f,s):
    out = str(datetime.datetime.now()) + '\t' + s
    print out
    f.write(out+'\n')