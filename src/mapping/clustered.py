import tensorflow as tf
import numpy as np

#Old version, only support cluster_num=k**2
def cluster_transform(sample, cluster_num=2):
    cluster_dim=int(np.log(cluster_num)/np.log(2))
    out_list=[]
    for dim in range(cluster_dim):
        inp=sample[:,dim:dim+1]
        out_list.append(tf.cast(tf.math.sign(inp), tf.float32)*tf.abs(inp)**0.2+inp) #!
    out_list.append(sample[:,cluster_dim:])
    return tf.concat(out_list, axis=1)

#Old version with learnable boundary
def cluster_transform_learnable(sample, cluster_num=2, boundary=0):
    cluster_dim=int(np.log(cluster_num)/np.log(2))
    out_list=[]
    for dim in range(cluster_dim):
        inp=sample[:,dim:dim+1]
        out_list.append(tf.cast(5*tf.math.sign(inp-boundary), tf.float32)*tf.abs(inp-boundary)**0.5+inp) #!
    out_list.append(sample[:,cluster_dim:])
    return tf.concat(out_list, axis=1)

#new version, support any integer cluster_num
def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(int(i))
    if n > 1:
        factors.append(int(n))
    return factors

def cluster_transform_general_single(sample, cluster_num=2):
    #Only for a single splitting g(y) = y + c1dis(y)c2r(ci(y))
    #Careful, this function is not safe at origin point
    angle_each=2*np.pi/cluster_num
    
    sample=tf.cast(sample, tf.float64)
    rci_matrix=[[np.cos(angle_each*(i+0.5)), np.sin(angle_each*(i+0.5))] for i in range(cluster_num)]
    rci_matrix=np.array(rci_matrix).astype(np.float64)
    
    
    l=tf.reduce_sum(sample**2, axis=1)**0.5
    
    angle=tf.math.acos(-sample[:,0]/l)
    sign=tf.math.sign(-sample[:,1])
    sign=sign-(sign-1)*(sign+1)
    angle=angle*sign
    angle+=np.pi
    
    half_cluster_angle=np.pi/cluster_num
    idx=tf.cast(angle//angle_each, tf.int64)
    
    rci=tf.gather(rci_matrix, idx, axis=0)
    dis_angle=half_cluster_angle-tf.abs(angle-angle_each*(tf.cast(idx, tf.float64)+0.5))
    
    dis=l*tf.math.sin(dis_angle)
    
    moving=tf.expand_dims(5*dis**0.5, axis=-1)*rci
    return tf.cast(sample+moving, tf.float32)
    

def cluster_transform_general(sample, cluster_num=2):
    #For multiply splitting
    factor_list=prime_factors(cluster_num)
    out_list=[]
    dim=0
    for i in range(len(factor_list)):
        inp=sample[:,dim:dim+2]
        out_list.append(cluster_transform_general_single(inp, cluster_num=2))
        dim+=2
    out_list.append(sample[:,dim:])
    return tf.concat(out_list, axis=1)
    

class Cluster(tf.keras.Model):
    def __init__(self, cluster_num):
        super(Cluster, self).__init__()
        self.cluster_num=cluster_num
        #br=tf.Variable(0.0, dtype=tf.float32)
        
        self.l=tf.keras.layers.Dense(1)
        #br=self.l(np.array([[0.0]]).astype(np.float32))[0,0]
        
        #br=0.0
        #self.br=br
        #br_exp=tf.math.exp(br)
        #self.boundary=4*(br_exp/(1+br_exp))-2
            
    def call(self, x):
        #x_mapped=cluster_transform(x, self.cluster_num)
        #x_mapped=cluster_transform_general(x, self.cluster_num)
        br=self.l(np.array([[0.0]]).astype(np.float32))[0,0]
        br_exp=tf.math.exp(br)
        #boundary=4*(br_exp/(1+br_exp))-2
        boundary=8*(br_exp/(1+br_exp))-4
        x_mapped=cluster_transform_learnable(x, self.cluster_num, boundary)
        return x_mapped, []  
    
    def load_weights(self, path):
        pass
    def save_weights(self, path):
        pass
        
if __name__=='__main__':
    cluster=Cluster(2)
    #a=np.random.random((20, 10)).astype(np.float32)
    a=np.array([[1,0.0001], [1,-0.0001],[0,1], [-1,0], [-2,0],[1e-7,1e-9]]).astype(np.float32)
    b=cluster(a)
    #print(a.shape, b.shape)
    print(cluster.boundary)