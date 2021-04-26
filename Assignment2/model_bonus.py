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
        self.train_cost = []
        self.val_cost = []
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
        for l in self.layers:
            l.x = X_c.copy()
            X_c = np.maximum(0,l.W @ l.x + l.b)

        return softmax(l.W @ l.x + l.b)

    def compute_gradients(self, X, Y, P):
        G = -(Y-P)
        X_c = X.copy()
        nb = X.shape[1]
        for l in reversed(self.layers):
            l.grad_W = 1/nb * G @ l.x.T + 2 * self.lambda_ * l.W
            l.grad_b = (1/nb * np.sum(G,axis=1)).reshape(l.d_out,1)
            G = l.W.T @ G
            G = np.multiply(G,np.heaviside(l.x,0))
        


    def update_params(self, eta):

        for k in range(0, self.k_layers):
            self.layers[k].W -= eta * self.layers[k].grad_W
            self.layers[k].b -= eta * self.layers[k].grad_b

    # def computeGradientsNum(self, X, Y, h=1e-5):
    #     grad_bs, grad_Ws = [], []

    #     for j in tqdm(range(self.k_layers)):
    #         grad_bs.append(np.zeros(self.layers[j].d_out))
    #         for i in range(self.layers[j].d_out):

    #             self.layers[j].b[i][0] -= h
    #             _,c1 = self.computeCost(X, Y)
    #             self.layers[j].b[i][0] += 2 * h
    #             _,c2 = self.computeCost(X, Y)

    #             self.layers[j].b[i][0] -= h
    #             grad_bs[j][i] = (c2 - c1) / (2*h)

    #     for j in tqdm(range(self.k_layers)):
    #         grad_Ws.append(np.zeros((self.layers[j].d_out, self.layers[j].d_in)))
    #         for i in range(self.layers[j].d_out):
    #             for l in range(self.layers[j].d_in):

    #                 self.layers[j].W[i, l] -= h
    #                 _,c1 = self.computeCost(X, Y)

    #                 self.layers[j].W[i, l] += 2*h
    #                 _,c2 = self.computeCost(X, Y)

                    
    #                 self.layers[j].W[i, l] -= h
    #                 grad_Ws[j][i, l] = (c2 - c1) / (2*h)

    #     return grad_Ws, grad_bs

    def computeGradientsNum(self, X_batch, Y_batch, h=1e-5):
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
                _, c1 = self.computeCost(X_batch, Y_batch)
                layer.b = np.copy(b_try)
                layer.b[i] -= h
                _, c2 = self.computeCost(X_batch, Y_batch)
                grads['b' + str(j)][i] = (c1-c2) / (2*h)
            layer.b = b_try

            W_try = np.copy(selfW)
            for i in np.ndindex(selfW.shape):
                layer.W = np.copy(W_try)
                layer.W[i] += h
                _, c1 = self.computeCost(X_batch, Y_batch)
                layer.W = np.copy(W_try)
                layer.W[i] -= h
                _, c2 = self.computeCost(X_batch, Y_batch)
                grads['W' + str(j)][i] = (c1-c2) / (2*h)
            layer.W = W_try

        return grads

    # def compareGradients(self, X, Y, eps=1e-10, h=1e-5):
    #     """ Compares analytical and numerical gradients given a certain epsilon """
    #     gn_Ws, gn_bs = self.computeGradientsNum(X, Y, h)
    #     rerr_w, rerr_b = [], []
    #     aerr_w, aerr_b = [], []

    #     for i in range(self.k_layers):
    #         rerr_w.append(rel_error(self.layers[i].grad_W, gn_Ws[i], eps))
    #         rerr_b.append(rel_error(self.layers[i].grad_b, gn_bs[i], eps))
    #         aerr_w.append(np.mean(abs(self.layers[i].grad_W - gn_Ws[i])))
    #         aerr_b.append(np.mean(abs(self.layers[i].grad_b - gn_bs[i])))

    #     return rerr_w, rerr_b, aerr_w, aerr_b

    def compareGradients(self, X, Y, eps=1e-10, h=1e-5):
        """ Compares analytical and numerical gradients given a certain epsilon """
        gn = self.computeGradientsNum(X, Y, h)
        rerr_w, rerr_b = [], []
        aerr_w, aerr_b = [], []

        def _rel_error(x, y, eps): return np.abs(
            x-y)/max(eps, np.abs(x)+np.abs(y))

        def rel_error(g1, g2, eps):
            vfunc = np.vectorize(_rel_error)
            return np.mean(vfunc(g1, g2, eps))

        for i, layer in enumerate(self.layers):
            rerr_w.append(rel_error(layer.grad_W, gn[f'W{i}'], eps))
            rerr_b.append(rel_error(layer.grad_b, gn[f'b{i}'], eps))
            aerr_w.append(np.mean(abs(layer.grad_W - gn[f'W{i}'])))
            aerr_b.append(np.mean(abs(layer.grad_b - gn[f'b{i}'])))

        return rerr_w, rerr_b, aerr_w, aerr_b

    def computeCost(self, X, Y):

        P = self.forward(X)
        loss = np.log(np.sum(np.multiply(Y, P), axis=0))
        loss = - np.sum(loss)/X.shape[1]
        r = np.sum([np.linalg.norm(layer.W) ** 2 for layer in self.layers])
        cost = loss + self.lambda_ * r
        return loss, cost

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

    def cyclicLearning(self,data, GDparams, experiment, build_ensemble,augment, verbose, save):

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
        cycle = 0

        for e in tqdm(range(epochs)):
            X_train, Y_train, y_train = shuffle(X_train.T, Y_train.T, y_train, random_state=e)
            X_train = X_train.T
            Y_train = Y_train.T
            for i in range(l):
                X_batch = X_train[:,i*(n_batch):(i+1)*n_batch].copy()
                Y_batch = Y_train[:,i*(n_batch):(i+1)*n_batch].copy()
                if augment:
                    r = np.random.rand()
                    X_batch =  augment_data(X_batch,r)

                p = self.forward(X_batch)
                self.compute_gradients(X_batch,Y_batch,p)
                self.update_params(eta)

                if t % (2*ns/freq) == 0:
                    self.history(data, t, verbose)
                
                if t <= ns:
                    eta = eta_min + t/ns * (eta_max - eta_min)
                else:
                    eta = eta_max - (t - ns)/ns * (eta_max - eta_min)
        
                t = (t+1) % (2*ns)
                if t == 0 and build_ensemble:

                    self.save({},GDparams, experiment,True, cycle=cycle)
                    cycle += 1


        train_loss_ = np.array(self.train_loss)
        val_loss_ = np.array(self.val_loss)
        train_acc_ = np.array(self.train_acc)
        val_acc_ = np.array(self.val_acc)
        train_cost_ = np.array(self.train_cost)
        val_cost_ = np.array(self.val_cost)
        hist = {'train_loss':train_loss_, 'val_loss':val_loss_, 'train_acc':train_acc_, 'val_acc':val_acc_,'val_cost':val_cost_,'train_cost':train_cost_}
        if save:
            self.save(hist,GDparams, experiment,True)

    def find_etas(self, data, GDparams, freq=20):

        X_train, Y_train, y_train = data["X_train"], data["Y_train"], data["y_train"]
        X_val, y_val =  data["X_val"], data["y_val"],

        n = X_train.shape[1]

        epochs = GDparams["epochs"]
        n_batch = GDparams["n_batch"]
        eta_min = GDparams["eta_min"]
        eta_max = GDparams["eta_max"]
        iterations = int(n/n_batch)
        delta_eta = (eta_max - eta_min) / (iterations * epochs) * freq
        eta = eta_min
        etas = [eta]


        v_acc = self.ComputeAccuracy(X_val, y_val)
        self.val_acc.append(v_acc)

        for epoch in tqdm(range(epochs)):

            X_train, Y_train, y_train = shuffle(X_train.T, Y_train.T, y_train.T, random_state=epoch)
            X_train, Y_train, y_train = X_train.T, Y_train.T, y_train.T

            for i in range(iterations):
                p = self.forward(X_train[:,i*(n_batch):(i+1)*n_batch])
                self.compute_gradients(X_train[:,i*(n_batch):(i+1)*n_batch],Y_train[:,i*(n_batch):(i+1)*n_batch],p)
                self.update_params(eta)

                if i % freq == 0:
                    eta += delta_eta
                    etas.append(eta)

                    v_acc = self.ComputeAccuracy(X_val, y_val)
                    self.val_acc.append(v_acc)
            
        hist = {'etas':etas, 'val_acc':self.val_acc}

        return hist

    def save(self, hist, GDparams, experiment,cyclic,cycle=-1):
        n_batch = GDparams['n_batch']
        if cyclic:
            eta_min, eta_max = GDparams['eta_min'], GDparams['eta_max']
            ns = GDparams["ns"]
            n_cycles = GDparams['n_cycles']
            freq = GDparams['freq']
            if (cycle >=0):
                np.save(f"Models/bonus/layers_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{self.lambda_}_{freq}_{self.seed}_{cycle}.npy",self.layers)
            else:
                np.save(f"Models/bonus/hist_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{self.lambda_}_{freq}_{self.seed}.npy",hist)
                np.save(f"Models/bonus/layers_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{self.lambda_}_{freq}_{self.seed}.npy",self.layers)
        else:
            epochs = GDparams['epochs']
            eta = GDparams['eta']
            np.save(f"Models/bonus/hist_{experiment}_{epochs}_{n_batch}_{eta}_{self.lambda_}_{self.seed}.npy",hist)
            np.save(f"Models/bonus/layers_{experiment}_{epochs}_{n_batch}_{eta}_{self.lambda_}_{self.seed}.npy",self.layers)


    def history(self,data, eps, verbose):
        X_train,Y_train,y_train = data['X_train'], data['Y_train'], data['y_train']
        X_val,Y_val,y_val = data['X_val'], data['Y_val'], data['y_val']
        t_loss, t_cost  = self.computeCost(X_train,Y_train)
        v_loss, v_cost = self.computeCost(X_val,Y_val)
        t_acc = self.ComputeAccuracy(X_train,y_train)
        v_acc = self.ComputeAccuracy(X_val,y_val)
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.train_cost.append(t_cost)
        self.val_cost.append(v_cost)
        self.train_acc.append(t_acc)
        self.val_acc.append(v_acc)
        if verbose:
            print(f"\t Epoch {eps}: train_cost = {t_cost}, val_cost = {v_cost},  \n \t train_acc = {t_acc}, val_acc = {v_acc}")

#####################################################################
# utilities

_rel_error = lambda x,y,eps: np.abs(x-y)/max(eps,np.abs(x)+np.abs(y))

def rel_error(g1, g2, eps):
    vfunc = np.vectorize(_rel_error)
    return np.mean(vfunc(g1,g2,eps))

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def augment_data(X, flip=0, sigma=1):
        X_new = np.copy(X) 
        noise = np.random.normal(0, sigma, (X.shape))
        X_new += noise
        mean, std = np.mean(X_new, axis=1), np.std(X_new, axis = 1)
        X_new -= np.outer(mean, np.ones(X_new.shape[1]))
        X_new /= np.outer(std, np.ones(X.shape[1]))
        d, n_batch = X.shape
        if flip > 0.5:
            X_new = X_new.reshape(n_batch, 3, 32, 32).transpose(0, 2, 3, 1)
            X_new = np.array([np.fliplr(X_new[i]) for i in range(n_batch)])
            X_new = X_new.reshape((d, n_batch))
        return X_new

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
            layers = np.load(f"Models/bonus/layers_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{lambda_}_{freq}_{seed}_{cycle}.npy",allow_pickle=True)
            hist = None
        else:
            hist = np.load(f"Models/bonus/hist_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{lambda_}_{freq}_{seed}.npy",allow_pickle=True)
            layers = np.load(f"Models/bonus/layers_{experiment}_{ns}_{n_cycles}_{n_batch}_{eta_min}_{eta_max}_{lambda_}_{freq}_{seed}.npy",allow_pickle=True)
    else:
        epochs = GDparams['epochs']
        eta = GDparams['eta']
        hist = np.load(f"Models/bonus/hist_{experiment}_{epochs}_{n_batch}_{eta}_{lambda_}_{seed}.npy",allow_pickle=True)
        layers = np.load(f"Models/bonus/layers_{experiment}_{epochs}_{n_batch}_{eta}_{lambda_}_{seed}.npy",allow_pickle=True)

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

def ensemble_classification(X_test,y_test,models):
    y = 20*np.ones(y_test.shape)
    predictions = np.zeros(y_test.shape)
    for model in models:
        X_c = X_test.copy()
        for layer in model:
            layer.x = X_c.copy()
            X_c = np.maximum(0,layer.W @ layer.x + layer.b)
        p = softmax(layer.W @ layer.x + layer.b)
        y_hat = np.argmax(p,axis=0)
        y = np.vstack((y,y_hat))
    for i,pred in enumerate(y.T):
        pred = np.array([int(p) for p in pred])
        counts = np.bincount(pred)
        predictions[i] = np.argmax(counts)
    acc = np.sum(predictions == y_test)
    return acc/len(y_test)
    


