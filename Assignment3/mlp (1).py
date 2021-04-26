from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from enum import Enum


class Initialization(Enum):
    XAVIER = 1
    HE = 2


def batch_normalize(X, mean, std):
    return (X - mean)/std


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    return np.maximum(0, x)


class Layer():
    def __init__(self, d_in, d_out, activation, init=Initialization.XAVIER):
        self.d_in = d_in
        self.d_out = d_out
        init = init.value if isinstance(init, Initialization) else init 
        self.W = np.random.normal(
            0, init/np.sqrt(d_in), (d_out, d_in))
        self.b = np.zeros((d_out, 1))
        self.activation = activation
        self.init = init
        self.input = None
        self.grad_W = None
        self.grad_b = None

    def evaluate_layer(self, input, train_mode=True, init=False):
        self.input = input.copy()
        return self.activation(self.W @ self.input + self.b)

    def compute_gradients(self, G, n_batch, lamda, propagate=False):
        self.grad_W = G @ self.input.T / n_batch + \
            2 * lamda * self.W
        self.grad_b = np.mean(G, axis=1, keepdims=True) 
        if propagate:
            G = self.W.T @ G
            G = np.multiply(G, np.heaviside(self.input, 0))
        return G

    def update_params(self, eta):
        self.W -= eta * self.grad_W
        self.b -= eta * self.grad_b


class BNLayer(Layer):
    def __init__(self, d_in, d_out, activation, init=Initialization.HE, alpha=0.9):
        super().__init__(d_in, d_out, activation, init)
        self.alpha = alpha
        self.mu = np.zeros((self.d_out, 1))
        self.v = np.zeros((self.d_out, 1))
        self.mu_av = np.zeros((self.d_out, 1))
        self.v_av = np.zeros((self.d_out, 1))
        self.gamma = np.ones((self.d_out, 1))
        self.beta = np.zeros((self.d_out, 1))
        self.grad_gamma = None
        self.grad_beta = None
        self.scores = None
        self.scores_hat = None

    def evaluate_layer(self, input, train_mode=True, init=False):
        self.input = input.copy()
        self.scores = self.W @ self.input + self.b

        if train_mode:
            self.mu = np.mean(self.scores, axis=1, keepdims=True)
            self.v = np.var(self.scores, axis=1, ddof=1, keepdims=True)

            if init:
                self.mu_av = self.mu
                self.v_av = self.v
            else:
                self.mu_av = self.alpha * self.mu_av + (1-self.alpha) * self.mu
                self.v_av = self.alpha * self.v_av + (1-self.alpha) * self.v

            self.scores_hat = batch_normalize(
                self.scores, self.mu, np.sqrt(self.v + np.finfo(float).eps))

        # test mode
        else:
            self.scores_hat = batch_normalize(
                self.scores, self.mu_av, np.sqrt(self.v_av + np.finfo(np.float64).eps))

        return self.activation(np.multiply(self.gamma, self.scores_hat) + self.beta)

    def compute_gradients(self, G, n_batch, lamda, propagate=False):
        self.grad_gamma = np.sum(np.multiply(
            G, self.scores_hat), axis=1, keepdims=True) / n_batch
        self.grad_beta = np.sum(G, axis=1, keepdims=True) / n_batch

        G = np.multiply(G, self.gamma)
        G = self.batch_norm_back_pass(G, n_batch)

        G = super().compute_gradients(G, n_batch, lamda, propagate=propagate)
        return G

    def batch_norm_back_pass(self, G, n_batch):

        sigma1 = np.power(self.v + np.finfo(np.float64).eps, -0.5)
        sigma2 = np.power(self.v + np.finfo(np.float64).eps, -1.5)

        G1 = np.multiply(G, sigma1)
        G2 = np.multiply(G, sigma2)

        D = self.scores - self.mu

        c = np.sum(np.multiply(G2, D), axis=1, keepdims=True)

        G = G1 - np.sum(G1, axis=1, keepdims=True) / \
            n_batch - np.multiply(D, c) / n_batch
        return G

    def update_params(self, eta):
        super().update_params(eta)
        self.gamma -= eta * self.grad_gamma
        self.beta -= eta * self.grad_beta


class MLP():
    def __init__(self, k=2, dims=[3072, 50, 10], lamda=0, seed=42, batch_norm=False, alpha=0.9, init=Initialization.HE):
        np.random.seed(seed)
        self.seed = seed
        self.k = k
        self.lamda = lamda
        self.dims = dims
        self.layers = []
        self.batch_norm = batch_norm
        self.add_layers(init, alpha)
        self.train_loss, self.val_loss = [], []
        self.train_cost, self.val_cost = [], []
        self.train_acc, self.val_acc = [], []

    def add_layers(self, init, alpha):
        for i in range(self.k):
            d_in, d_out = self.dims[i], self.dims[i+1]
            activation = relu if i < self.k-1 else softmax
            if self.batch_norm and i < self.k-1:
                layer = BNLayer(d_in, d_out, activation,
                                alpha=alpha, init=init)
            else:
                layer = Layer(d_in, d_out, activation, init)
            self.layers.append(layer)

    def forward_pass(self, X, train_mode=True, init=False):
        input = X.copy()
        for layer in self.layers:
            input = layer.evaluate_layer(input, train_mode, init)
        return input

    def compute_cost(self, X, Y, train_mode=True, init=False):
        """ Computes the cost function: cross entropy loss + L2 regularization """
        P = self.forward_pass(X, train_mode, init)
        loss = np.log(np.sum(np.multiply(Y, P), axis=0))
        loss = - np.sum(loss)/X.shape[1]
        r = np.sum([np.linalg.norm(layer.W) ** 2 for layer in self.layers])
        cost = loss + self.lamda * r
        return loss, cost

    def compute_gradients(self, X, Y, P):
        G = - (Y - P)
        n_batch = X.shape[1]
        for i, layer in enumerate(reversed(self.layers)):
            G = layer.compute_gradients(
                G, n_batch, self.lamda, propagate=(i != self.k-1))

    def update_parameters(self, eta=1e-2):
        for layer in self.layers:
            layer.update_params(eta)

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
                _, c1 = self.compute_cost(X_batch, Y_batch)
                layer.b = np.copy(b_try)
                layer.b[i] -= h
                _, c2 = self.compute_cost(X_batch, Y_batch)
                grads['b' + str(j)][i] = (c1-c2) / (2*h)
            layer.b = b_try

            W_try = np.copy(selfW)
            for i in np.ndindex(selfW.shape):
                layer.W = np.copy(W_try)
                layer.W[i] += h
                _, c1 = self.compute_cost(X_batch, Y_batch)
                layer.W = np.copy(W_try)
                layer.W[i] -= h
                _, c2 = self.compute_cost(X_batch, Y_batch)
                grads['W' + str(j)][i] = (c1-c2) / (2*h)
            layer.W = W_try

            if isinstance(layer, BNLayer):
                selfGamma = layer.gamma
                selfBeta = layer.beta
                grads['gamma' + str(j)] = np.zeros(selfBeta.shape)
                grads['beta' + str(j)] = np.zeros(selfGamma.shape)

                gamma_try = np.copy(selfGamma)
                for i in range(selfGamma.shape[0]):
                    layer.gamma = np.copy(gamma_try)
                    layer.gamma[i] += h
                    _, c1 = self.compute_cost(X_batch, Y_batch)
                    layer.gamma = np.copy(gamma_try)
                    layer.gamma[i] -= h
                    _, c2 = self.compute_cost(X_batch, Y_batch)
                    grads['gamma' + str(j)][i] = (c1-c2) / (2*h)
                layer.gamma = gamma_try

                beta_try = np.copy(selfBeta)
                for i in range(selfBeta.shape[0]):
                    layer.beta = np.copy(beta_try)
                    layer.beta[i] += h
                    _, c1 = self.compute_cost(X_batch, Y_batch)
                    layer.beta = np.copy(gamma_try)
                    layer.beta[i] -= h
                    _, c2 = self.compute_cost(X_batch, Y_batch)
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
            if isinstance(layer, BNLayer):
                rerr_gamma.append(
                    rel_error(layer.grad_gamma, gn[f'gamma{i}'], eps))
                rerr_beta.append(
                    rel_error(layer.grad_beta, gn[f'beta{i}'], eps))
                aerr_gamma.append(
                    fun(abs(layer.grad_gamma - gn[f'gamma{i}'])))
                aerr_beta.append(
                    fun(abs(layer.grad_beta - gn[f'beta{i}'])))

        if self.batch_norm:
            return rerr_w, aerr_w, rerr_b, aerr_b, rerr_gamma, aerr_gamma, rerr_beta, aerr_beta
        else:
            return rerr_w, aerr_w, rerr_b, aerr_b

    def compute_accuracy(self, X, y, train_mode=False):
        """ Computes the prediction accuracy of a given state of the network """
        P = self.forward_pass(X, train_mode=train_mode)
        y_pred = np.argmax(P, axis=0)
        return accuracy_score(y, y_pred)

    def mini_batch_gd(self, data, GDparams, verbose=True, backup=False):
        """ Performas minibatch gradient descent """

        X, Y, y = data["X_train"], data["Y_train"], data["y_train"]
        _, n = X.shape

        epochs, batch_size, eta = GDparams["n_epochs"], GDparams["n_batch"], GDparams["eta"]
        self.history(data, 0, verbose, cyclic=False)

        for epoch in tqdm(range(epochs)):

            X, Y, y = shuffle(X.T, Y.T, y.T, random_state=epoch)
            X, Y, y = X.T, Y.T, y.T

            for j in range(n//batch_size):
                j_start = j * batch_size
                j_end = (j+1) * batch_size
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                P_batch = self.forward_pass(X_batch, train_mode=True, init=(epoch==0 and j==0))

                self.compute_gradients(X_batch, Y_batch, P_batch)

                self.update_parameters(eta)

            self.history(data, epoch, verbose, cyclic=False)

        if backup:
            self.backup(GDparams)

    def cyclic_learning(self, data, GDparams, verbose=True, backup=False):
        """ Performas minibatch gradient descent """
        X, Y, y = data["X_train"], data["Y_train"], data["y_train"]

        _, n = X.shape

        n_cycles, batch_size, eta_min, eta_max, ns, freq = GDparams["n_cycles"], GDparams[
            "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"], GDparams['freq']

        eta = eta_min
        t = 0

        epochs = batch_size * 2 * ns * n_cycles // n

        for epoch in tqdm(range(epochs)):

            X, Y, y = shuffle(X.T, Y.T, y.T, random_state=epoch)
            X, Y, y = X.T, Y.T, y.T

            for j in range(n//batch_size):
                j_start = j * batch_size
                j_end = (j+1) * batch_size
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                P_batch = self.forward_pass(
                    X_batch, train_mode=True, init=(epoch == 0 & j == 0))

                self.compute_gradients(X_batch, Y_batch, P_batch)
                self.update_parameters(eta)

                if t % (2*ns//freq) == 0:
                    self.history(data, t, verbose)

                if t <= ns:
                    eta = eta_min + t/ns * (eta_max - eta_min)
                else:
                    eta = eta_max - (t - ns)/ns * (eta_max - eta_min)

                t = (t+1) % (2*ns)
        if backup:
            self.backup_cyclic(GDparams)

    def history(self, data, epoch, verbose=True, cyclic=True):
        """ Creates history of the training """

        X, Y, y, X_val, Y_val, y_val = data["X_train"], data["Y_train"], data[
            "y_train"], data["X_val"], data["Y_val"], data["y_val"]

        t_loss, t_cost = self.compute_cost(X, Y, train_mode=False)
        v_loss, v_cost = self.compute_cost(X_val, Y_val, train_mode=False)

        t_acc = self.compute_accuracy(X, y, train_mode=False)
        v_acc = self.compute_accuracy(X_val, y_val, train_mode=False)

        if verbose:
            pref = "Update Step " if cyclic else "Epoch "
            print(
                f'{pref}{epoch}: train_acc={t_acc} | val_acc={v_acc} | train_loss={t_loss} | val_loss={v_loss} | train_cost={t_cost} | val_cost={v_cost}')

        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.train_cost.append(t_cost)
        self.val_cost.append(v_cost)
        self.train_acc.append(t_acc)
        self.val_acc.append(v_acc)

    def backup(self, GDparams):
        """ Saves networks params in order to be able to reuse it """

        epochs, batch_size, eta, exp = GDparams["n_epochs"], GDparams["n_batch"], GDparams["eta"], GDparams["exp"]

        np.save(
            f'History/{exp}_layers_{epochs}_{batch_size}_{eta}_{self.lamda}_{self.seed}.npy', self.layers)

        hist = {"train_loss": self.train_loss, "train_acc": self.train_acc, "train_cost": self.train_cost,
                "val_loss": self.val_loss, "val_acc": self.val_acc, "val_cost": self.val_cost}

        np.save(
            f'History/{exp}_hist_{epochs}_{batch_size}_{eta}_{self.lamda}_{self.seed}.npy', hist)

    def backup_cyclic(self, GDparams):
        """ Saves networks params in order to be able to reuse it for cyclic learning"""

        n_cycles, batch_size, eta_min, eta_max, ns, exp = GDparams["n_cycles"], GDparams[
            "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"], GDparams["exp"]

        np.save(
            f'History/{exp}_layers_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.npy', self.layers)

        hist = {"train_loss": self.train_loss, "train_acc": self.train_acc, "train_cost": self.train_cost,
                "val_loss": self.val_loss, "val_acc": self.val_acc, "val_cost": self.val_cost}

        np.save(
            f'History/{exp}_hist_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.npy', hist)

    def plot_metric(self, GDparams, metric="loss", cyclic=True):
        """ Plots a given metric (loss or accuracy) """

        if cyclic:
            n_cycles, batch_size, eta_min, eta_max, ns = GDparams["n_cycles"], GDparams[
                "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"]
        else:
            epochs, batch_size, eta = GDparams["n_epochs"], GDparams["n_batch"], GDparams["eta"]

        batch_size, exp = GDparams["n_batch"], GDparams['exp']

        if metric == "loss":
            plt.ylim(1, 4)
            plt.plot(self.train_loss, label=f"Train {metric}")
            plt.plot(self.val_loss, label=f"Validation {metric}")
        elif metric == "accuracy":
            plt.ylim(0, np.max(self.train_acc)+0.1)
            plt.plot(self.train_acc, label=f"Train {metric}")
            plt.plot(self.val_acc, label=f"Validation {metric}")
        else:
            plt.ylim(1, 6)
            plt.plot(self.train_cost, label=f"Train {metric}")
            plt.plot(self.val_cost, label=f"Validation {metric}")

        plt.xlabel("epochs")
        plt.ylabel(metric)
        if cyclic:
            plt.title(f"Monitoring of {metric} during {n_cycles} cycles.")
        else:
            plt.title(f"Monitoring of {metric} during {epochs} epochs.")
        plt.legend()
        if cyclic:
            plt.savefig(
                f'History/{exp}_{metric}_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.png')
        else:
            plt.savefig(
                f'History/{exp}_{metric}_{epochs}_{batch_size}_{eta}_{self.lamda}_{self.seed}.png')
        plt.show()

    @staticmethod
    def load_mlp(GDparams, cyclic=True, k=2, dims=[3072, 50, 10], lamda=0, seed=42, batch_norm=True, init=Initialization.HE):
        mlp = MLP(k, dims, lamda, seed, batch_norm=batch_norm, init=init)
        if cyclic:

            n_cycles, batch_size, eta_min, eta_max, ns, exp = GDparams["n_cycles"], GDparams[
                "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"], GDparams["exp"]
            layers = np.load(
                f'History/{exp}_layers_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{mlp.lamda}_{mlp.seed}.npy', allow_pickle=True)
            hist = np.load(
                f'History/{exp}_hist_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{mlp.lamda}_{mlp.seed}.npy', allow_pickle=True)
        else:

            epochs, batch_size, eta, exp = GDparams["n_epochs"], GDparams[
                "n_batch"], GDparams["eta"], GDparams["exp"]

            layers = np.load(
                f'History/{exp}_layers_{epochs}_{batch_size}_{eta}_{mlp.lamda}_{mlp.seed}.npy', allow_pickle=True)

            hist = np.load(
                f'History/{exp}_hist_{epochs}_{batch_size}_{eta}_{mlp.lamda}_{mlp.seed}.npy', allow_pickle=True)

        mlp.layers = layers

        mlp.train_acc = hist.item()['train_acc']
        mlp.train_loss = hist.item()["train_loss"]
        mlp.train_cost = hist.item()["train_cost"]
        mlp.val_acc = hist.item()['val_acc']
        mlp.val_loss = hist.item()["val_loss"]
        mlp.val_cost = hist.item()["val_cost"]

        return mlp


class Search():
    
    def __init__(self, l_min=-5, l_max=-1, n_lambda=20, sample=True, seed=42):
        np.random.seed(seed)
        self.l_min = l_min
        self.l_max = l_max
        self.n_lambda = n_lambda
        if sample:
            self.lambdas = self.sample_lambda()
        else: 
            self.lambdas = np.linspace(l_min, l_max, num=n_lambda)

    def sample_lambda(self):
        exp = self.l_min + (self.l_max - self.l_min) * \
            np.random.rand(self.n_lambda)
        lambdas = [10**e for e in exp]
        return lambdas

    def random_search(self, data, GDparams, lamdas=None, k=3, dims=[3072,50,50,10], batch_norm=True, init=Initialization.HE):
        if lamdas is not None:
            self.lambdas = lamdas
        for lmda in self.lambdas:
            mlp = MLP(lamda=lmda, k=k, dims=dims, batch_norm=batch_norm, init=init)
            mlp.cyclic_learning(
                data, GDparams, verbose=False, backup=True)

    def random_search_perf(self, GDparams, lamdas=None, k=3, dims=[3072,50,50,10], batch_norm=True, init=Initialization.HE):
        if lamdas is not None:
            self.lambdas = lamdas
        models = defaultdict(list)
        for lmda in self.lambdas:
            model = MLP.load_mlp(GDparams, cyclic=True, lamda=lmda,
                                 k=k, dims=dims, batch_norm=batch_norm, init=init)
            models[model.val_acc[-1]*100
                ].append({"lamda": round(lmda,7), "train_acc": round(model.train_acc[-1]*100, 5)})
        for acc in sorted(models.keys(), reverse=True):
            for v in models[acc]:
                print(f'{v["lamda"]} & {v["train_acc"]} & {round(acc,5)} \\\\')
