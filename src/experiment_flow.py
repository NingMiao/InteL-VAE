import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt
from PIL import Image
import os

from src.mapping.sparse import sparse_metric, sparse_loss_function
from src.misc import make_recursive_dir

#Util Functions
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

def KL_term(mean_1, log_var_1, mean_2, log_var_2):
    dim_KL = 0.5*(log_var_2-log_var_1)+(tf.exp(log_var_1)+(mean_1-mean_2)**2)/(2*tf.exp(log_var_2))-0.5
    return tf.reduce_sum(dim_KL, axis=1)

def reparameterize(mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean
  
def reparameterize_multi(mean, logvar, num=10):
    shape=mean.shape
    shape_new=[mean.shape[0], num]
    for i in range(1, len(shape)):
        shape_new.append(shape[i])
    eps = tf.random.normal(shape=shape_new)   
    return eps * tf.exp(tf.expand_dims(logvar, axis=1) * .5) + tf.expand_dims(mean, axis=1)

def laplace(x, b=0.1):
    return tf.abs(x)

def der_tanh(x):
    return 1 - tf.math.tanh(x) ** 2

tanh=tf.math.tanh

def Sylvester_flow(zk, r1, r2, q_ortho, b, sum_ldj=True):
    """
    All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
    outside of this function. Computes the following transformation:
    z' = z + QR1 h( R2Q^T z + b)
    or actually
    z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
    :param zk: shape: (batch_size, z_size)
    :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
    :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
    :param q_ortho: shape (batch_size, z_size , num_ortho_vecs)
    :param b: shape: (batch_size, 1, self.z_size)
    :return: z, log_det_j
    """

    # Amortized flow parameters
    zk=tf.expand_dims(zk, axis=1)

    # Save diagonals for log_det_j
    diag_r1=tf.linalg.diag_part(r1)
    diag_r2=tf.linalg.diag_part(r2)

    r1_hat = r1
    r2_hat = r2

    qr2 = tf.matmul(q_ortho, tf.transpose(r2_hat, [0,2, 1]))
    qr1 = tf.matmul(q_ortho, r1_hat)
    
    
    r2qzb = tf.matmul(zk, qr2) + b
    z = tf.matmul(tanh(r2qzb), tf.transpose(qr1, [0,2, 1])) + zk
    z = z[:,0]
   
    # Compute log|det J|
    # Output log_det_j in shape (batch_size) instead of (batch_size,1)
    diag_j = diag_r1 * diag_r2
    diag_j = tf.expand_dims(der_tanh(r2qzb), 1) * diag_j
    diag_j += 1.
    log_diag_j = tf.math.log(tf.abs(diag_j))

    if sum_ldj:
        log_det_j = tf.reduce_sum(log_diag_j, axis=-1)
    else:
        log_det_j = log_diag_j
    return z, log_det_j


class Flow_Experiment(tf.keras.Model):
    def __init__(self, model, batch_size, mapping=None, experiment_id='flow_experiment'):
        super(Flow_Experiment, self).__init__()
        
        self.model=model
        self.mapping=mapping
        self.batch_size=batch_size
        self.experiment_id=experiment_id
        #For generation
        self.random_vector_for_generation = tf.random.normal(shape=[16, self.model.latent_dim])
        
        #For training
        with tf.device('/device:GPU:0'):
            self.optimizer = tf.keras.optimizers.Adam()
        
        #Record
        self.reconstruction_logprob_record = tf.keras.metrics.Mean(name='reconstruction')
        self.KL_term_record = tf.keras.metrics.Mean(name='KL')        
        self.ELBO_record = tf.keras.metrics.Mean(name='ELBO')
        self.loss_record = tf.keras.metrics.Mean(name='loss')
        self.logP_record = tf.keras.metrics.Mean(name='logP')
        
        ##Flow
        
        self.z_size=self.model.latent_dim
        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        self.num_flows = 4
        self.num_ortho_vecs = 8

        # Orthogonalization parameters
        self.cond = 1.e-6

        self.steps = 30
        identity = tf.eye(self.num_ortho_vecs, self.num_ortho_vecs)
        # Add batch dimension
        self._eye = tf.expand_dims(identity, 0)

        # Masks needed for triangular R1 and R2.
        triu_mask = tf.linalg.band_part(tf.ones([self.num_ortho_vecs, self.num_ortho_vecs], dtype=tf.float32), 0, -1)
        self.triu_mask = tf.expand_dims(tf.expand_dims(triu_mask,0),3)

        # Amortized flow parameters
        # Diagonal elements of R1 * R2 have to satisfy -1 < R1 * R2 for flow to be invertible
        self.diag_activation = tf.keras.layers.Activation('tanh')
        
        self.amor_d = tf.keras.layers.Dense(units=self.num_flows * self.num_ortho_vecs * self.num_ortho_vecs, activation=None)

        self.amor_diag1 = tf.keras.layers.Dense(units=self.num_flows * self.num_ortho_vecs, activation=self.diag_activation)
        
        self.amor_diag2 = tf.keras.layers.Dense(units=self.num_flows * self.num_ortho_vecs, activation=self.diag_activation)

        self.amor_q = tf.keras.layers.Dense(units=self.num_flows * self.z_size * self.num_ortho_vecs, activation=None)
        self.amor_b = tf.keras.layers.Dense(self.num_flows * self.num_ortho_vecs, activation=None)
        
        self.debug_flag=0
        
    
    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size * num_flows, z_size * num_ortho_vecs)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, z_size, num_ortho_vecs)
        """

        # Reshape to shape (num_flows * batch_size, z_size * num_ortho_vecs)
        q = tf.reshape(q, [-1, self.z_size * self.num_ortho_vecs])

        norm = tf.norm(q, ord=2, axis=1, keepdims=True)
        amat = q/norm
        dim0 = amat.shape[0]
        amat = tf.reshape(amat, [dim0, self.z_size, self.num_ortho_vecs])

        max_norm = 0.

        # Iterative orthogonalization
        for s in range(self.steps):
            tmp = tf.matmul(tf.transpose(amat, [0, 2, 1]), amat)
            tmp = self._eye - tmp
            tmp = self._eye + 0.5 * tmp
            amat = tf.matmul(amat, tmp)

            # Testing for convergence
            test = tf.matmul(tf.transpose(amat, [0, 2, 1]), amat) - self._eye
            norms2 = tf.reduce_sum(tf.norm(test, ord=2, axis=2) ** 2, axis=1)
            norms = norms2**0.5
            max_norm = tf.reduce_max(norms)


        # Reshaping: first dimension is batch_size
        amat = tf.reshape(amat, [-1, self.num_flows, self.z_size, self.num_ortho_vecs])
        amat = tf.transpose(amat, [1,0,2,3])

        return amat
    
    def flow(self, z_mu, z_var):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        batch_size = z_mu.shape[0]

        # Amortized r1, r2, q, b for all flows

        full_d = self.amor_d(z_mu)
        diag1 = self.amor_diag1(z_mu)
        diag2 = self.amor_diag2(z_mu)

        full_d = tf.reshape(full_d, [batch_size, self.num_ortho_vecs, self.num_ortho_vecs, self.num_flows])
        diag1 = tf.reshape(diag1, [batch_size, self.num_ortho_vecs, self.num_flows])
        diag2 = tf.reshape(diag2, [batch_size, self.num_ortho_vecs, self.num_flows])

        r1 = full_d * self.triu_mask
        r2 = tf.transpose(full_d, [0, 2, 1, 3]) * self.triu_mask
        
        r1=tf.transpose(r1, [0,3,1,2])
        r1=tf.linalg.set_diag(r1, tf.transpose(diag1, [0,2,1]))
        r1=tf.transpose(r1, [0,2,3,1])
        r2=tf.transpose(r2, [0,3,1,2])
        r2=tf.linalg.set_diag(r2, tf.transpose(diag2, [0,2,1]))
        r2=tf.transpose(r2, [0,2,3,1])

        q = self.amor_q(z_mu)
        b = self.amor_b(z_mu)

        # Resize flow parameters to divide over K flows
        b = tf.reshape(b, [batch_size, 1, self.num_ortho_vecs, self.num_flows])

       
        """
        Flow with orthogonal sylvester flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        self.log_det_j = 0.

        # Orthogonalize all q matrices
        q_ortho = self.batch_construct_orthogonal(q)

        # Sample z_0
        z = [reparameterize(z_mu, z_var)]
        # Normalizing flows
        for k in range(self.num_flows):

            z_k, log_det_jacobian = Sylvester_flow(z[k], r1[:, :, :, k], r2[:, :, :, k], q_ortho[k, :, :, :], b[:, :, :, k])

            z.append(z_k)
            self.log_det_j += log_det_jacobian
        
        return z_mu, z_var, self.log_det_j, z[0], z[-1]

    
    @tf.function
    def compute_loss(self, x, beta, gamma, loss_type='se', training=False, IWAE=0):
        mean, logvar = self.model.encode(x, training=training)
        z_mu, z_var, ldj, z_0, z_K = self.flow(mean, logvar)
        
        if self.mapping:
            z_K, info=self.mapping(z_K)
        x_logit = self.model.decode(z_K, training=training)
  
        if loss_type=='cross_entropy':
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
            logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        elif loss_type=='se':
            logpx_z=-tf.reduce_sum((x_logit-x)**2, axis=[1,2,3])
        elif loss_type=='laplace':
            logpx_z=-tf.reduce_sum(laplace(x_logit-x), axis=[1,2,3])
  

        reconstruction = logpx_z
        #loss = -tf.reduce_mean(reconstruction) #!experiment
        loss_info = []
        
        
        ##New

        batch_size = x.shape[0]

        # - N E_q0 [ ln p(x|z_k) ]
        bce = - logpx_z

        # ln p(z_k)  (not averaged)
        log_p_zk = log_normal_pdf(z_K, 0.0, 0.0, raxis=1)
        # ln q(z_0)  (not averaged)
        log_q_z0 = log_normal_pdf(z_0, z_mu, z_var, raxis=1)
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        summed_logs = log_q_z0 - log_p_zk

        # sum over batches
        summed_ldj = ldj

        # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
        KL = (summed_logs - summed_ldj)
        elbo = reconstruction - KL*beta
        loss = -tf.reduce_mean(elbo)
        return loss, elbo, KL, reconstruction, loss_info

    @tf.function
    def compute_apply_gradients(self, x, beta, gamma, loss_type='se', IWAE=0):
        with tf.GradientTape() as tape:
            loss, elbo, KL, reconstruction, loss_info = self.compute_loss(x, beta, gamma, loss_type=loss_type, training=True, IWAE=IWAE)
        trainable_variables=self.trainable_variables
        if self.mapping:
            trainable_variables+=self.mapping.trainable_variables
        
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
  
        self.loss_record(loss)
        self.ELBO_record(elbo)
        self.KL_term_record(KL)
        self.reconstruction_logprob_record(reconstruction)
    
    @tf.function
    def compute_log_P(self, x, num=10):
        pass
  
    ##Quality check:
    def generate_and_save_images(self, epoch):
        folder='generate_images/'+self.experiment_id+'/'
        make_recursive_dir(folder)
        
        z = self.random_vector_for_generation
        if self.mapping:
            z, _ =self.mapping(z)
        predictions = self.model.decode(z)
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.model.image_shape[-1]==1:
                plt.imshow(predictions[i, :, :, 0], cmap='gray')
            else:
                img=predictions[i, :, :]
                plt.imshow(img)
            plt.axis('off')
        plt.savefig(folder+'/generate_image_at_epoch_{:04d}.png'.format(epoch))

    def reconstruct_and_save_images(self, epoch, test_dataset):
        folder='reconstruct_images/'+self.experiment_id+'/'
        make_recursive_dir(folder)
        
        x=[]
        i=0
        for batch in test_dataset:
            x.append(batch)
            i+=batch.shape[0]
            if i>=16:
                break
        x=np.concatenate(x, axis=0)[:16]
  
        for i in range(x.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.model.image_shape[-1]==1:
                plt.imshow(x[i, :, :, 0], cmap='gray')
            else:
                img=x[i, :, :]
                plt.imshow(img)
            plt.axis('off')
            plt.savefig(folder+'/image_real.png'.format(epoch))

        mean, logvar = self.model.encode(x)
        if self.mapping:
            mean, _=self.mapping(mean)
        
        predictions = self.model.decode(mean, apply_sigmoid=False)
  
        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            if self.model.image_shape[-1]==1:
                plt.imshow(predictions[i, :, :, 0], cmap='gray')
            else:
                img=predictions[i, :, :]
                plt.imshow(img)
            plt.axis('off')
        plt.savefig(folder+'/reconstruct_image_at_epoch_{:04d}.png'.format(epoch))
        
    def generate_and_save_images_for_FID(self, sample_size, folder=''):
        if folder=='':
            folder='FID_images/'+self.experiment_id+'/'
        make_recursive_dir(folder)
        try:
            os.system('rm '+folder+'*')
        except:
            pass
        sample_list=[]
        for i in range(sample_size//self.batch_size+1):
            test_input= np.random.normal(0,1,[self.batch_size, self.model.latent_dim]).astype(np.float32)
            if self.mapping:
                test_input, info=self.mapping(test_input)
            predictions = self.model.decode(test_input)
            sample_list.append(predictions)
        samples=np.concatenate(sample_list, axis=0)
        ##Some issue about clipping
        if self.model.image_shape[-1]==1:
            samples_max=np.max(samples, axis=(1,2,3), keepdims=True)
            samples_min=np.min(samples, axis=(1,2,3), keepdims=True)
            samples=(samples-samples_min)/(samples_max-samples_min)
            samples_color=np.tile(samples, [1,1,1,3])
        else:
            samples_color=samples
        for i in range(samples_color.shape[0]):
            img=Image.fromarray((samples_color[i]*255).astype(np.uint8))
            img.save(folder+str(i)+'.png')
        return folder
    
    
    #Run Experiment on datasets
    def train_epoch(self, train_dataset, beta, gamma, loss_type='se', IWAE=0):
        #Reset states
        start_time = time()
        self.loss_record.reset_states()
        self.ELBO_record.reset_states()
        self.KL_term_record.reset_states()
        self.reconstruction_logprob_record.reset_states()
        for train_x in train_dataset:
            self.compute_apply_gradients(train_x, beta, gamma, loss_type=loss_type, IWAE=IWAE)
        end_time = time()
        return self.loss_record.result().numpy(), self.ELBO_record.result().numpy(), self.KL_term_record.result().numpy(), self.reconstruction_logprob_record.result().numpy(), end_time-start_time, []
        
  
    def test_epoch(self, test_dataset, beta, gamma, loss_type='se', IWAE=0):
        #Reset states
        start_time = time()
        self.loss_record.reset_states()
        self.ELBO_record.reset_states()
        self.KL_term_record.reset_states()
        self.reconstruction_logprob_record.reset_states()
        for test_x in test_dataset:
            loss, elbo, KL, reconstruction , _ = self.compute_loss(test_x, beta, gamma, loss_type=loss_type, IWAE=IWAE)
            self.loss_record(loss)
            self.ELBO_record(elbo)
            self.KL_term_record(KL)
            self.reconstruction_logprob_record(reconstruction)
        end_time = time()
        return self.loss_record.result().numpy(), self.ELBO_record.result().numpy(), self.KL_term_record.result().numpy(), self.reconstruction_logprob_record.result().numpy(), end_time-start_time, []
    
    #Run Experiment on datasets 
    def sparse_score_epoch(self, test_dataset, test_size=2000):
        test_images=[]
        i=0
        for batch in test_dataset:
            test_images.append(batch)
            i+=batch.shape[0]
            if i>=test_size:
                break
        test_images=np.concatenate(test_images, axis=0)[:test_size]
        mean, logvar = self.model.encode(test_images)
        z = reparameterize(mean, logvar)
        if self.mapping:
            z_mapped, info=self.mapping(z)
        else:
            z_mapped=z
        return sparse_metric(z_mapped)  
    
    def logP_epoch(self, test_dataset_for_logP, num, sample_size):
        self.logP_record.reset_states()
        i=0
        for test_x in test_dataset_for_logP:
            self.compute_log_P(test_x, num)
            i+=test_x.shape[0]
            if i>=sample_size:
                break
        return self.logP_record.result()
 
    
    #Save and Load experiments
    def save(self, epoch=None):
        model_name=self.experiment_id
        if epoch:
            model_name=model_name+'_'+str(epoch)
        
        self.model.save_weights('model/'+model_name+'/CVAE/model.ckpt')
        if self.mapping:
            self.mapping.save_weights('model/'+model_name+'/mapping/model.ckpt')
        
    def restore(self, epoch=None):
        model_name=self.experiment_id
        if epoch:
            model_name=model_name+'_'+str(epoch)
            
        self.model.load_weights('model/'+model_name+'/CVAE/model.ckpt')
        if self.mapping:
            self.mapping.load_weights('model/'+model_name+'/mapping/model.ckpt')  
            
if __name__=='__main__':
    from src.CVAE import CVAE
    model=CVAE()
    ex=Flow_Experiment(model, 20, None)
    x=tf.random.uniform([100, 28, 28])
    ex.compute_apply_gradients(x, 1.0, 0.0, loss_type='se', IWAE=0)