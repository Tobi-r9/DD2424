import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from six.moves import cPickle
import pickle
from tqdm import tqdm
from collections import namedtuple
import utilis as u
from sklearn.utils import shuffle


class Layer():

    def __init__(self, d_in, d_out, W, b, grad_W, grad_b, x, alpha=0.9):
        self.d_in = d_in
        self.d_out = d_out
        self.W = W
        self.b = b
        self.grad_W = grad_W
        self.grad_b = grad_b
        self.x = x
        self.mu = np.zeros((self.d_out, 1))
        self.sigma = np.zeros((self.d_out, 1))
        self.mu_avg = None
        self.sigma_avg = None
        self.alpha = alpha
        self.gamma = np.ones((self.d_out, 1))
        self.grad_gamma = 0
        self.beta = np.zeros((self.d_out, 1))
        self.grad_beta = 0
        self.s1 = 0
        self.s2 = 0
        self.eps = np.finfo(np.float64).eps
        

    def batchnorm(self, s1,trainMode):
        n = self.x.shape[1]

        if trainMode:
            self.mu = np.mean(s1,axis=1, keepdims=True)
            self.sigma = np.var(s1, axis=1, keepdims=True) * ((n-1)/n)
            s2 = (s1 - self.mu)/np.sqrt(self.sigma +self.eps)
            s2 = np.multiply(self.gamma,s2) + self.beta

            #if not initialized before
            if self.mu_avg is None:
                self.mu_avg = self.alpha * self.mu + (1-self.alpha) * self.mu
                self.sigma_avg = self.alpha * self.sigma + (1-self.alpha) * self.sigma

            else:
                self.mu_avg = self.alpha * self.mu_avg + (1-self.alpha) * self.mu
                self.sigma_avg = self.alpha * self.sigma_avg + (1-self.alpha) * self.sigma

        #test time
        else:
            s2 = (s1 - self.mu_avg)/np.sqrt(self.sigma_avg +self.eps)
            s2 = np.multiply(self.gamma,s2) + self.beta
        return s2


    def BatchNomBackPass(self,G):
        n = self.x.shape[1]
        sigma1 = np.power(self.sigma + self.eps,-0.5)
        sigma2 = np.power(self.sigma + self.eps,-1.5)
        G1 = np.multiply(G,sigma1)
        G2 = np.multiply(G,sigma2)
        D = self.s1 - self.mu
        c = np.sum(np.multiply(G2,D),axis=1,keepdims=True)
        G = G1 - 1/n * np.sum(G1,axis=1,keepdims=True) - 1/n * np.multiply(D,c)
        return G

class MLP():

    def __init__(self, dimensions=[3072,50,10], lambda_=0, seed=42):
        np.random.seed(seed)
        self.seed = seed
        self.dimensions = dimensions
        self.k_layers = len(dimensions)-1
        self.lambda_ = lambda_
        self.layers = []
        self.init_layers()
        self.train_loss = []
        self.val_loss = []
        self.train_cost = []
        self.val_cost = []
        self.train_acc = []
        self.val_acc = []

    def init_layers(self):
        for k in range(self.k_layers):
            d_in = self.dimensions[k]
            d_out = self.dimensions[k+1]
            W = np.random.normal(0,2/np.sqrt(d_in),(d_out, d_in))
            b = np.zeros((d_out,1))
            grad_W = np.zeros((d_out, d_in))
            grad_b = np.zeros((d_out,1))
            x = 0
            layer = Layer(d_in, d_out, W, b, grad_W, grad_b, x)

            self.layers.append(layer)
        
       
    def forward(self, X):

        X_c = X.copy()
        for l in self.layers:
            l.x = X_c.copy()
            X_c = np.maximum(0,l.W @ l.x + l.b)

        return softmax(l.W @ l.x + l.b)

    

    def forward_batchnorm(self, X, trainMode=True):

        X_c = X.copy()
        for i,layer in enumerate(self.layers):
            layer.x = X_c.copy()
            if i != len(self.layers)-1:
                s1 = layer.W @ layer.x + layer.b
                s2 = layer.batchnorm(s1,trainMode)
                X_c = np.maximum(0,s2)

                if trainMode:
                    layer.s1 = s1
                    layer.s2 = s2
            
        return softmax(layer.W @ layer.x + layer.b)

    def compute_gradients(self, X, Y, P):
        
        G = -(Y-P)
        X_c = X.copy()
        nb = X.shape[1]
        for l in reversed(self.layers):
            l.grad_W = 1/nb * G @ l.x.T + 2 * self.lambda_ * l.W
            l.grad_b = (1/nb * np.sum(G,axis=1)).reshape(l.d_out,1)
            G = l.W.T @ G
            G = np.multiply(G,np.heaviside(l.x,0))
        
    def compute_gradients_batchnorm(self, X, Y, P):
        G = -(Y-P)
        X_c = X.copy()
        nb = X.shape[1]
        for i, layer in enumerate(reversed(self.layers)):
            if i != 0:
                layer.grad_gamma = (1/nb) * np.sum(np.multiply(G,layer.s2),axis=1,keepdims=True)
                layer.grad_beta = (1/nb) * np.sum(G,axis=1,keepdims=True)
                G = np.multiply(G,layer.gamma)
                G = layer.BatchNomBackPass(G)

            layer.grad_W = 1/nb * G @ layer.x.T + 2 * self.lambda_ * layer.W
            layer.grad_b = np.mean(G,axis=1,keepdims=True)
            G = layer.W.T @ G
            G = np.multiply(G,np.heaviside(layer.x,0))
            

    def update_params(self, eta):

        for i,layer in enumerate(self.layers)-1:
            layer.W -= eta * layer.grad_W
            layer.b -= eta * layer.grad_b

    def update_params_batchNorm(self,eta):

        for i,layer in enumerate(self.layers):
            layer.W -= eta * layer.grad_W
            layer.b -= eta * layer.grad_b
            if i != len(self.layers)-1:
                layer.gamma -= eta * layer.grad_gamma
                layer.beta -= eta * layer.grad_beta

                

    def compute_gradients_num(self, X_batch, Y_batch, h=1e-5):
        """ Numerically computes the gradients of the weight and bias parameters
        Args:
            X_batch (np.ndarray): data batch matrix (n_dims, n_samples)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (n_classes, n_samples)
            h            (float): marginal offset
        Returns:
            grad_W  (np.ndarray): the gradient of the weight parameter
            grad_b  (np.ndarray): the gradient of the bias parameter
        """
        grads = {}
        for j, layer in enumerate(self.layers):
            selfW = layer.W
            selfB = layer.b
            grads['W' + str(j)] = np.zeros(selfW.shape)
            grads['b' + str(j)] = np.zeros(selfB.shape)

            b_try = np.copy(selfB)
            for i in range(selfB.shape[0]):
                layer.b = np.copy(b_try)
                layer.b[i] += h
                _, c1 = self.computeCost(X_batch, Y_batch,True)
                layer.b = np.copy(b_try)
                layer.b[i] -= h
                _, c2 = self.computeCost(X_batch, Y_batch,True)
                grads['b' + str(j)][i] = (c1-c2) / (2*h)
            layer.b = b_try

            W_try = np.copy(selfW)
            for i in np.ndindex(selfW.shape):
                layer.W = np.copy(W_try)
                layer.W[i] += h
                _, c1 = self.computeCost(X_batch, Y_batch,True)
                layer.W = np.copy(W_try)
                layer.W[i] -= h
                _, c2 = self.computeCost(X_batch, Y_batch,True)
                grads['W' + str(j)][i] = (c1-c2) / (2*h)
            layer.W = W_try


            selfGamma = layer.gamma
            selfBeta = layer.beta
            grads['gamma' + str(j)] = np.zeros(selfBeta.shape)
            grads['beta' + str(j)] = np.zeros(selfGamma.shape)

            gamma_try = np.copy(selfGamma)
            for i in range(selfGamma.shape[0]):
                layer.gamma = np.copy(gamma_try)
                layer.gamma[i] += h
                _, c1 = self.computeCost(X_batch, Y_batch,True)
                layer.gamma = np.copy(gamma_try)
                layer.gamma[i] -= h
                _, c2 = self.computeCost(X_batch, Y_batch,True)
                grads['gamma' + str(j)][i] = (c1-c2) / (2*h)
            layer.gamma = gamma_try

            beta_try = np.copy(selfBeta)
            for i in range(selfBeta.shape[0]):
                layer.beta = np.copy(beta_try)
                layer.beta[i] += h
                _, c1 = self.computeCost(X_batch, Y_batch,True)
                layer.beta = np.copy(gamma_try)
                layer.beta[i] -= h
                _, c2 = self.computeCost(X_batch, Y_batch,True)
                grads['beta' + str(j)][i] = (c1-c2) / (2*h)
            layer.beta = beta_try

        return grads

    def compare_gradients(self, X, Y, eps=1e-10, h=1e-5, fun=np.mean):
        """ Compares analytical and numerical gradients given a certain epsilon """
        gn = self.compute_gradients_num(X, Y, h)
        rerr_w, rerr_b, rerr_gamma, rerr_beta = [], [], [], []
        aerr_w, aerr_b, aerr_gamma, aerr_beta = [], [], [], []

        def _rel_error(x, y, eps): return np.abs(
            x-y)/max(eps, np.abs(x)+np.abs(y))

        def rel_error(g1, g2, eps):
            vfunc = np.vectorize(_rel_error)
            return fun(vfunc(g1, g2, eps))

        for i, layer in enumerate(self.layers):
            rerr_w.append(rel_error(layer.grad_W, gn[f'W{i}'], eps))
            rerr_b.append(rel_error(layer.grad_b, gn[f'b{i}'], eps))
            aerr_w.append(fun(abs(layer.grad_W - gn[f'W{i}'])))
            aerr_b.append(fun(abs(layer.grad_b - gn[f'b{i}'])))

            rerr_gamma.append(
                rel_error(layer.grad_gamma, gn[f'gamma{i}'], eps))
            rerr_beta.append(
                rel_error(layer.grad_beta, gn[f'beta{i}'], eps))
            aerr_gamma.append(
                fun(abs(layer.grad_gamma - gn[f'gamma{i}'])))
            aerr_beta.append(
                fun(abs(layer.grad_beta - gn[f'beta{i}'])))

        return rerr_w, aerr_w, rerr_b, aerr_b, rerr_gamma, aerr_gamma, rerr_beta, aerr_beta



    def computeCost(self, X, Y, batchNorm):
        if batchNorm:
            P = self.forward_batchnorm(X,False)
        else:
            P = self.forward(X)

        loss = np.log(np.sum(np.multiply(Y, P), axis=0))
        loss = - np.sum(loss)/X.shape[1]
        r = np.sum([np.linalg.norm(layer.W) ** 2 for layer in self.layers])
        cost = loss + self.lambda_ * r
        return loss, cost

    def ComputeAccuracy(self,X,y, batchNorm):
        if batchNorm:
            p = self.forward_batchnorm(X,False)
        else:
            p = self.forward(X)

        y_hat = np.argmax(p,axis=0)
        acc = np.sum(y_hat == y)
        return acc/len(y)


    def cyclicLearning(self,data, GDparams, batchNorm, experiment, verbose, save):

        X_train,Y_train,y_train = data['X_train'], data['Y_train'], data['y_train']
        X_val,Y_val,y_val = data['X_val'], data['Y_val'], data['y_val']
        
        freq = GDparams['freq']
        eta_min, eta_max = GDparams['eta_min'], GDparams['eta_max']
        ns = GDparams["ns"]
        n_batch = GDparams['n_batch']
        n_cycles = GDparams['n_cycles']
        epochs = int((n_batch * 2 * ns * n_cycles)/X_train.shape[1])
        l = int(X_train.shape[1]/n_batch)

        # self.history(X, Y, y,  X_val, Y_val, y_val, 0, verbose)

        eta = eta_min
        t = 0

        for e in tqdm(range(epochs)):
            X_train, Y_train, y_train = shuffle(X_train.T, Y_train.T, y_train, random_state=e)
            X_train = X_train.T
            Y_train = Y_train.T
            for i in range(l):
                
                if batchNorm:
                    p = self.forward_batchnorm(X_train[:,i*(n_batch):(i+1)*n_batch])
                    self.compute_gradients_batchnorm(X_train[:,i*(n_batch):(i+1)*n_batch],Y_train[:,i*(n_batch):(i+1)*n_batch],p)
                    self.update_params_batchNorm(eta)
                else:
                    p = self.forward(X_train[:,i*(n_batch):(i+1)*n_batch])
                    self.compute_gradients(X_train[:,i*(n_batch):(i+1)*n_batch],Y_train[:,i*(n_batch):(i+1)*n_batch],p)
                    self.update_params(eta)

                if t % (2*ns/freq) == 0:
                    self.history(data, t, verbose, batchNorm)
                
                if t <= ns:
                    eta = eta_min + t/ns * (eta_max - eta_min)
                else:
                    eta = eta_max - (t - ns)/ns * (eta_max - eta_min)
        
                t = (t+1) % (2*ns)


        train_loss_ = np.array(self.train_loss)
        val_loss_ = np.array(self.val_loss)
        train_acc_ = np.array(self.train_acc)
        val_acc_ = np.array(self.val_acc)
        train_cost_ = np.array(self.train_cost)
        val_cost_ = np.array(self.val_cost)
        hist = {'train_loss':train_loss_, 'val_loss':val_loss_, 'train_acc':train_acc_, 'val_acc':val_acc_,'train_cost':train_cost_,'val_cost':val_cost_}
        if save:
            self.save(hist,GDparams, experiment,True)

        return hist

    def save(self, hist, GDparams, experiment,cyclic):
        n_batch = GDparams['n_batch']
        if cyclic:
            eta_min, eta_max = GDparams['eta_min'], GDparams['eta_max']
            ns = GDparams["ns"]
            n_cycles = GDparams['n_cycles']
            freq = GDparams['freq']
            
            np.save(f"Models/hist_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{self.lambda_}_{freq}_{self.seed}.npy",hist)
            np.save(f"Models/layers_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{self.lambda_}_{freq}_{self.seed}.npy",self.layers)
        else:
            epochs = GDparams['epochs']
            eta = GDparams['eta']
            np.save(f"Models/hist_{experiment}_{epochs}_{n_batch}_{eta}_{self.lambda_}_{self.seed}.npy",hist)
            np.save(f"Models/layers_{experiment}_{epochs}_{n_batch}_{eta}_{self.lambda_}_{self.seed}.npy",self.layers)


    def history(self,data, eps, verbose, batchNorm):
        X_train,Y_train,y_train = data['X_train'], data['Y_train'], data['y_train']
        X_val,Y_val,y_val = data['X_val'], data['Y_val'], data['y_val']
        t_loss, t_cost  = self.computeCost(X_train,Y_train,batchNorm)
        v_loss, v_cost = self.computeCost(X_val,Y_val,batchNorm)
        t_acc = self.ComputeAccuracy(X_train,y_train,batchNorm)
        v_acc = self.ComputeAccuracy(X_val,y_val,batchNorm)
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.train_cost.append(t_cost)
        self.val_cost.append(v_cost)
        self.train_acc.append(t_acc)
        self.val_acc.append(v_acc)
        if verbose:
            print(f"\t Epoch {eps}: train_cost = {t_cost}, val_cost = {v_cost},  \n \t train_acc = {t_acc}, val_acc = {v_acc}")

_rel_error = lambda x,y,eps: np.abs(x-y)/max(eps,np.abs(x)+np.abs(y))


def rel_error(g1, g2, eps):
    vfunc = np.vectorize(_rel_error)
    return np.mean(vfunc(g1,g2,eps))

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class LambdaSearch():

    def __init__(self, l_min, l_max,n_lambda):
        self.l_min = l_min
        self.l_max = l_max
        self.n_lambda = n_lambda
        self.lambdas = []
        self.models = {}

    def sample_lambda(self):
        intervall = np.linspace(self.l_min,self.l_max,self.n_lambda)
        self.lambdas = [10**(i) for i in intervall]

    def lambda_search(self,data,GDparams):

        self.sample_lambda()
        for lmda in self.lambdas:
            mlp = MLP(lambda_=lmda)
            hist = mlp.cyclicLearning(data, GDparams, 'lambda_search', False, False)
            self.models.update({np.max(mlp.val_acc):mlp})

        return self.models


    def grid_search(self, data, GDparams, lmda):

        for param1 in self.params1:
            for param2 in self.params2:
                GDparams[self.p1] = param1
                GDparams[self.p2] = param2
                mlp = MLP(lambda_=lmda)
                hist = mlp.cyclicLearning(data, GDparams, 'grid_search', False, False)
                self.models.update({mlp.val_acc[-1]:mlp})

        

def load_network(GDparams,experiment,cyclic,cycle=-1):
    seed = GDparams['seed']
    n_batch = GDparams['n_batch']
    lambda_ = GDparams['lambda']
    if cyclic:
        eta_min, eta_max = GDparams['eta_min'], GDparams['eta_max']
        ns = GDparams["ns"]
        n_cycles = GDparams['n_cycles']
        freq = GDparams['freq']

        if cycle >=0:
            layers = np.load(f"Models/layers_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{lambda_}_{freq}_{seed}_{cycle}.npy",allow_pickle=True)
            hist = None
        else:
            hist = np.load(f"Models/hist_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{lambda_}_{freq}_{seed}.npy",allow_pickle=True)
            layers = np.load(f"Models/layers_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{lambda_}_{freq}_{seed}.npy",allow_pickle=True)
    else:
        epochs = GDparams['epochs']
        eta = GDparams['eta']
        hist = np.load(f"Models/hist_{experiment}_{epochs}_{n_batch}_{eta}_{lambda_}_{seed}.npy",allow_pickle=True)
        layers = np.load(f"Models/layers_{experiment}_{epochs}_{n_batch}_{eta}_{lambda_}_{seed}.npy",allow_pickle=True)

    return layers, hist

def get_test_acc(X_test,y_test,layers):
    X_c = X_test.copy()
    for l in layers:
        l.x = X_c.copy()
        X_c = np.maximum(0,l.W @ l.x + l.b)
    p = softmax(l.W @ l.x + l.b)
    y_hat = np.argmax(p,axis=0)
    acc = np.sum(y_hat == y_test)
    return acc/len(y_test)
