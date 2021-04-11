import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from six.moves import cPickle
import pickle
from tqdm import tqdm
from collections import namedtuple
import utilis as u
from sklearn.utils import shuffle

Layer = namedtuple('Layer', ['d_in', 'd_out', 'W',
                             'b', 'grad_W', 'grad_b', 'X'])

class Layer():

    def __init__(self, d_in, d_out, W, b, grad_W, grad_b, x):
        self.d_in = d_in
        self.d_out = d_out
        self.W = W
        self.b = b
        self.grad_W = grad_W
        self.grad_b = grad_b
        self.x = x

class MLP():

    def __init__(self, dimensions=[3072,50,10], k_layers=2, lambda_=0, seed=42):
        np.random.seed(seed)
        self.seed = seed
        self.dimensions = dimensions
        self.k_layers = k_layers
        self.lambda_ = lambda_
        self.layers = []
        self.init_layers()
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def init_layers(self):
        for k in range(self.k_layers):
            d_in = self.dimensions[k]
            d_out = self.dimensions[k+1]
            W = np.random.normal(0,1/np.sqrt(d_in),(d_out, d_in))
            b = np.zeros((d_out,1))
            grad_W = np.zeros((d_out, d_in))
            grad_b = np.zeros((d_out,1))
            x = 0
            layer = Layer(d_in, d_out, W, b, grad_W, grad_b, x)

            self.layers.append(layer)
        
       

    def forward(self, X):

        X_c = X.copy()
        for k in range(self.k_layers):
            X_c = np.maximum(0,self.layers[k].W @ X_c + self.layers[k].b)
            self.layers[k].x = X_c

        return softmax(X_c)

    def compute_gradients(self, X, Y, P):
        G = -(Y-P)
        X_c = X.copy()
        nb = X.shape[1]
        for k in range(self.k_layers-1,0,-1):
            self.layers[k].grad_W = 1/nb * G @ self.layers[k-1].x.T + 2 * self.lambda_ * self.layers[k].W
            self.layers[k].grad_b = 1/nb * np.sum(G,axis=1)
            G = self.layers[k].W.T @ G
            G = np.multiply(G,np.heaviside(self.layers[k-1].x,0))
        
        self.layers[0].grad_W = 1/nb * G @ X_c.T +  2 * self.lambda_ * self.layers[0].W
        self.layers[0].grad_b = 1/nb * np.sum(G,axis=1)


    def update_params(self, eta):

        for k in range(0, self.k_layers):
            self.layers[k].W -= eta * self.layers[k].grad_W
            self.layers[k].b[:,0] -= eta * self.layers[k].grad_b

    def computeGradientsNum(self, X, Y, h=1e-7):
        grad_bs, grad_Ws = [], []

        for j in tqdm(range(self.k_layers)):
            grad_bs.append(np.zeros(self.layers[j].d_out))
            for i in range(self.layers[j].d_out):

                #self.layers[j].b[i] -= h
                layers = self.layers
                layers[j].b[i] -= h
                c1 = self.computeCost(X, Y, layers)

                #self.layers[j].b[i] += 2 * h
                layers = self.layers
                layers[j].b[i] += h
                c2 = self.computeCost(X, Y, layers)

                #self.layers[j].b[i] -= h

                grad_bs[j][i] = (c2 - c1) / (2*h)

        for j in tqdm(range(self.k_layers)):
            grad_Ws.append(np.zeros((self.layers[j].d_out, self.layers[j].d_in)))
            for i in range(self.layers[j].d_out):
                for l in range(self.layers[j].d_in):
                    layers = self.layers
                    layers[j].W[i, l] -= h 
                    c1 = self.computeCost(X, Y, layers)

                    #self.layers[j].W[i, l] += 2*h
                    layers = self.layers
                    layers[j].W[i, l] += h 
                    c2 = self.computeCost(X, Y, layers)

                    
                    #self.layers[j].W[i, l] -= h

                    grad_Ws[j][i, l] = (c2 - c1) / (2*h)

        return grad_Ws, grad_bs

    def compareGradients(self, X, Y, eps=1e-10, h=1e-7):
        """ Compares analytical and numerical gradients given a certain epsilon """
        gn_Ws, gn_bs = self.computeGradientsNum(X, Y, h)
        rerr_w, rerr_b = [], []
        aerr_w, aerr_b = [], []

        for i in range(self.k_layers):
            rerr_w.append(rel_error(self.layers[i].grad_W, gn_Ws[i], eps))
            rerr_b.append(rel_error(self.layers[i].grad_b, gn_bs[i], eps))
            aerr_w.append(np.mean(abs(self.layers[i].grad_W - gn_Ws[i])))
            aerr_b.append(np.mean(abs(self.layers[i].grad_b - gn_bs[i])))

        return rerr_w, rerr_b, aerr_w, aerr_b

    def computeCost(self, X, Y):
        """ Computes the cost function: cross entropy loss + L2 regularization """
        P = self.forward(X)
        l = -np.log(np.sum(np.multiply(Y, P), axis=0))
        r = np.sum([np.linalg.norm(self.layers[i].W)
                    **2 for i in range(self.k_layers)])
        J = np.sum(l)/X.shape[1] + self.lambda_ * r
        return J

    def ComputeAccuracy(self,X,y):
        p = self.forward(X)
        y_hat = np.argmax(p,axis=0)
        acc = np.sum(y_hat == y)
        return acc/len(y)

    def MiniBatch(self,data,GDparams,experiment,verbose,save):
        
        X_train,Y_train,y_train = data['X_train'], data['Y_train'], data['y_train']
        X_val,Y_val,y_val = data['X_val'], data['Y_val'], data['y_val']

        eta = GDparams['eta']
        n_batch = GDparams['n_batch']
        epochs = GDparams['epochs']
        l = int(X_train.shape[1]/n_batch)

        self.history(data,0,verbose)

        for e in tqdm(range(epochs)):
            X_train, Y_train, y_train = shuffle(X_train.T, Y_train.T, y_train, random_state=e)
            X_train = X_train.T
            Y_train = Y_train.T

            for i in range(l):
                p = self.forward(X_train[:,i*(n_batch):(i+1)*n_batch])
                self.compute_gradients(X_train[:,i*(n_batch):(i+1)*n_batch],Y_train[:,i*(n_batch):(i+1)*n_batch],p)
                self.update_params(eta)
            
            self.history(data,e+1,verbose)

        train_loss_ = np.array(self.train_loss)
        val_loss_ = np.array(self.val_loss)
        train_acc_ = np.array(self.train_acc)
        val_acc_ = np.array(self.val_acc)
        

        hist = {'train_loss':train_loss_, 'val_loss':val_loss_, 'train_acc':train_acc_, 'val_acc':val_acc_}
        if save:
            self.save(hist,GDparams,experiment,False)

        return hist

    def cyclicLearning(self,data, GDparams, experiment, verbose, save):

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

                p = self.forward(X_train[:,i*(n_batch):(i+1)*n_batch])
                self.compute_gradients(X_train[:,i*(n_batch):(i+1)*n_batch],Y_train[:,i*(n_batch):(i+1)*n_batch],p)
                self.update_params(eta)

                if t % (2*ns/freq) == 0:
                    self.history(data, t, verbose)
                
                if t <= ns:
                    eta = eta_min + t/ns * (eta_max - eta_min)
                else:
                    eta = eta_max - (t - ns)/ns * (eta_max - eta_min)
        
                t = (t+1) % (2*ns)


        train_loss_ = np.array(self.train_loss)
        val_loss_ = np.array(self.val_loss)
        train_acc_ = np.array(self.train_acc)
        val_acc_ = np.array(self.val_acc)
        hist = {'train_loss':train_loss_, 'val_loss':val_loss_, 'train_acc':train_acc_, 'val_acc':val_acc_}
        if save:
            self.save(hist,GDparams, experiment,True)

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

    def plot_history(self,hist,metric):

        if metric == 'loss':
            plt.plot(hist['train_loss'])
            plt.plot(hist['val_loss'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['train_loss', 'val_loss'])
        elif metric == 'acc':
            plt.plot(hist['train_acc'])
            plt.plot(hist['val_acc'])
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.legend(['train_acc', 'val_acc'])
        plt.show()


    def history(self,data, eps, verbose):
        X_train,Y_train,y_train = data['X_train'], data['Y_train'], data['y_train']
        X_val,Y_val,y_val = data['X_val'], data['Y_val'], data['y_val']
        t_loss = self.computeCost(X_train,Y_train)
        v_loss = self.computeCost(X_val,Y_val)
        t_acc = self.ComputeAccuracy(X_train,y_train)
        v_acc = self.ComputeAccuracy(X_val,y_val)
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.train_acc.append(t_acc)
        self.val_acc.append(v_acc)
        if verbose:
            print(f"\t Epoch {eps}: train_loss = {t_loss}, val_loss = {v_loss},  \n \t train_acc = {t_acc}, val_acc = {v_acc}")

_rel_error = lambda x,y,eps: np.abs(x-y)/max(eps,np.abs(x)+np.abs(y))


def rel_error(g1, g2, eps):
    vfunc = np.vectorize(_rel_error)
    return np.mean(vfunc(g1,g2,eps))

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Search():

    def __init__(self, l_min, l_max,n_lambda, p1,params1,p2,params2):
        self.l_min = l_min
        self.l_max = l_max
        self.params1 = params1
        self.params2 = params2
        self.p1 = p1
        self.p2 = p2
        self.n_lambda = n_lambda
        self.lambdas = []
        self.models = {}

    def sample_lambda(self):
        r = self.l_min + (self.l_max - self.l_min)*np.random.rand(self.n_lambda[0])
        self.lambdas = [10**i for i in r]

    def random_search(self,data,GDparams):

        self.sample_lambda()
        for t, num in enumerate(self.n_lambda):
            for lmda in self.lambdas:

                if self.params1 and self.params2:
                    self.grid_search(data, GDparams, lmda)

                else:
                    mlp = MLP(lambda_=lmda)
                    hist = mlp.cyclicLearning(data, GDparams, 'lambda_search', False, False)
                    self.models.update({mlp.val_acc[-1]:mlp})
            try:
                n = self.n_lambda[t+1]
            except:
                n = 3
            self.update_lambda(n=n)

        max_key = max(self.models.keys())
        return self.models[max_key]


    def grid_search(self, data, GDparams, lmda):

        for param1 in self.params1:
            for param2 in self.params2:
                GDparams[self.p1] = param1
                GDparams[self.p2] = param2
                mlp = MLP(lambda_=lmda)
                hist = mlp.cyclicLearning(data, GDparams, 'grid_search', False, False)
                self.models.update({mlp.val_acc[-1]:mlp})


    def update_lambda(self,n,_min=1e-2,_max=1e-2):
        key = max(self.models.keys())
        lba = self.models[key].lambda_
        l_min_ = lba-_min
        l_max_ = lba+_max
        r = l_min_ + ((l_max_ - l_min_)*np.random.rand(n))
        self.lambdas = [lmbda for lmbda in r]
        

    


