from itertools import cycle
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from collections import namedtuple
from utils import softmax
from tqdm import tqdm
from collections import Counter

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
    def __init__(self, k=2, dims=[3072, 50, 10], lamda=0, seed=42) -> None:
        np.random.seed(seed)
        self.seed = seed
        self.k = k
        self.lamda = lamda
        self.dims = dims
        self.layers = []
        for i in range(k):
            d_in, d_out = self.dims[i], self.dims[i+1]
            self.layers.append(Layer(d_in, d_out, np.random.normal(
                0, 1/np.sqrt(d_in), (d_out, d_in)), np.zeros((d_out, 1)), None, None, None))

        self.train_loss, self.val_loss = [], []
        self.train_cost, self.val_cost = [], []
        self.train_acc, self.val_acc = [], []

    def forwardpass(self, X):
        Xc = X.copy()
        for i in range(self.k):
            self.layers[i].x = np.maximum(
                0, self.layers[i].W @ Xc + self.layers[i].b)
            Xc = self.layers[i].x.copy()

        return softmax(self.layers[-1].x)

    def computeCost(self, X, Y):
        """ Computes the cost function: cross entropy loss + L2 regularization """
        P = self.forwardpass(X)
        loss = -np.log(np.sum(np.multiply(Y, P), axis=0))
        loss = np.sum(loss)/X.shape[1]
        r = np.sum([np.linalg.norm(self.layers[i].W)
                    ** 2 for i in range(self.k)])
        cost = loss + self.lamda * r
        return loss, cost

    def computeGradients(self, X, Y, P):
        G = - (Y - P)
        nb = X.shape[1]

        for i in range(self.k-1, 0, -1):
            self.layers[i].grad_W = G @ self.layers[i-1].x.T / \
                nb + 2 * self.lamda * self.layers[i].W
            self.layers[i].grad_b = (
                np.sum(G, axis=1) / nb).reshape(self.layers[i].d_out, 1)
            G = self.layers[i].W.T @ G
            G = np.multiply(G, np.heaviside(self.layers[i-1].x, 0))

        self.layers[0].grad_W = G @ X.T / nb + \
            2 * self.lamda * self.layers[0].W
        self.layers[0].grad_b = (
            np.sum(G, axis=1) / nb).reshape(self.layers[0].d_out, 1)

    def updateParameters(self, eta=1e-2):
        for i in range(self.k):
            self.layers[i].W -= eta * self.layers[i].grad_W
            self.layers[i].b -= eta * self.layers[i].grad_b

    def computeGradientsNum(self, X, Y, h=1e-5):
        grad_bs, grad_Ws = [], []

        for j in tqdm(range(self.k)):
            grad_bs.append(np.zeros(self.layers[j].d_out))
            for i in range(self.layers[j].d_out):
                self.layers[j].b[i] -= h
                _, c1 = self.computeCost(X, Y)

                self.layers[j].b[i] += 2 * h
                _, c2 = self.computeCost(X, Y)

                self.layers[j].b[i] -= h

                grad_bs[j][i] = (c2 - c1) / (2*h)

        for j in tqdm(range(self.k)):
            grad_Ws.append(
                np.zeros((self.layers[j].d_out, self.layers[j].d_in)))
            for i in range(self.layers[j].d_out):
                for l in range(self.layers[j].d_in):
                    self.layers[j].W[i, l] -= h
                    _, c1 = self.computeCost(X, Y)

                    self.layers[j].W[i, l] += 2*h
                    _, c2 = self.computeCost(X, Y)

                    self.layers[j].W[i, l] -= h

                    grad_Ws[j][i, l] = (c2 - c1) / (2*h)

        return grad_Ws, grad_bs

    def compareGradients(self, X, Y, eps=1e-10, h=1e-5):
        """ Compares analytical and numerical gradients given a certain epsilon """
        gn_Ws, gn_bs = self.computeGradientsNum(X, Y, h)
        rerr_w, rerr_b = [], []
        aerr_w, aerr_b = [], []

        def _rel_error(x, y, eps): return np.abs(
            x-y)/max(eps, np.abs(x)+np.abs(y))

        def rel_error(g1, g2, eps):
            vfunc = np.vectorize(_rel_error)
            return np.mean(vfunc(g1, g2, eps))

        for i in range(self.k):
            rerr_w.append(rel_error(self.layers[i].grad_W, gn_Ws[i], eps))
            rerr_b.append(rel_error(self.layers[i].grad_b, gn_bs[i], eps))
            aerr_w.append(np.mean(abs(self.layers[i].grad_W - gn_Ws[i])))
            aerr_b.append(np.mean(abs(self.layers[i].grad_b - gn_bs[i])))

        return rerr_w, rerr_b, aerr_w, aerr_b

    def computeAccuracy(self, X, y):
        """ Computes the prediction accuracy of a given state of the network """
        P = self.forwardpass(X)
        y_pred = np.argmax(P, axis=0)
        return accuracy_score(y, y_pred)

    def minibatchGD(self, X, Y, y, X_val, Y_val, y_val, GDparams, verbose=True, experiment="test_gradients", backup=False):
        """ Performas minibatch gradient descent """
        _, n = X.shape
        epochs, batch_size, eta = GDparams["n_epochs"], GDparams[
            "n_batch"], GDparams["eta_min"]

        self.history(X, Y, y,  X_val, Y_val, y_val, 0, verbose)

        for epoch in tqdm(range(epochs)):

            X, Y, y = shuffle(X.T, Y.T, y.T, random_state=epoch)
            X, Y, y = X.T, Y.T, y.T

            for j in range(n//batch_size):
                j_start = j * batch_size
                j_end = (j+1) * batch_size
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                P_batch = self.forwardpass(X_batch)

                self.computeGradients(X_batch, Y_batch, P_batch)

                self.updateParameters(eta)

            self.history(X, Y, y,  X_val, Y_val, y_val, epoch, verbose)
        
        if backup:
            self.backup(GDparams, experiment=experiment)

    def cyclicLearning(self, X, Y, y, X_val, Y_val, y_val, GDparams, verbose=True, experiment="test_gradients", backup=False, freq=10):
        """ Performas minibatch gradient descent """
        _, n = X.shape
        epochs, batch_size, eta_min, eta_max, ns = GDparams["n_epochs"], GDparams[
            "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"]

        eta = eta_min
        t, c = 0, 0

        for _ in tqdm(range(epochs)):
            for j in range(n//batch_size):
                j_start = j * batch_size
                j_end = (j+1) * batch_size
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                P_batch = self.forwardpass(X_batch)

                self.computeGradients(X_batch, Y_batch, P_batch)
                self.updateParameters(eta)

                if t % (2*ns/freq) == 0:
                    self.history(X, Y, y,  X_val, Y_val, y_val, t, verbose)
                
                if t <= ns:
                    eta = eta_min + t/ns * (eta_max - eta_min)
                else:
                    eta = eta_max - (t - ns)/ns * (eta_max - eta_min)
        
                t = (t+1) % (2*ns)
                if t == 0 and experiment=="ensemble_learning":
                    if verbose:
                        print(f"Cycle {c} saved")
                    self.backup(GDparams, experiment=experiment, cycle=c)
                    c += 1
        if backup:
            self.backup(GDparams, experiment=experiment)

    def history(self, X, Y, y, X_val, Y_val, y_val, epoch, verbose=True):
        """ Creates history of the training """
        t_loss, t_cost = self.computeCost(X, Y)
        v_loss, v_cost = self.computeCost(X_val, Y_val)

        t_acc = self.computeAccuracy(X, y)
        v_acc = self.computeAccuracy(X_val, y_val)

        if verbose:
            print(
                f'Update Step {epoch}: train_acc={t_acc} | val_acc={v_acc} | train_loss={t_loss} | val_loss={v_loss} | train_cost={t_cost} | val_cost={v_cost}')
        
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.train_cost.append(t_cost)
        self.val_cost.append(v_cost)
        self.train_acc.append(t_acc)
        self.val_acc.append(v_acc)

    def backup(self, GDparams, experiment="test_gradients", cycle=0):
        """ Saves networks params in order to be able to reuse it """

        epochs, batch_size, eta_min, eta_max, ns = GDparams["n_epochs"], GDparams[
            "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"]

        np.save(
            f'History/{experiment}_layers_{cycle}_{epochs}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.npy', self.layers)

        hist = {"train_loss": self.train_loss, "train_acc": self.train_acc, "train_cost": self.train_cost,
                "val_loss": self.val_loss, "val_acc": self.val_acc, "val_cost": self.val_cost}

        np.save(
            f'History/{experiment}_hist_{cycle}_{epochs}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.npy', hist)

    def plot_metric(self, GDparams, metric="loss", experiment="test_gradients"):
        """ Plots a given metric (loss or accuracy) """
        epochs, batch_size, eta_min, eta_max, ns = GDparams["n_epochs"], GDparams[
            "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"]
        
        if metric == "loss":
            plt.ylim(0, 3)
            plt.plot(self.train_loss, label=f"Train {metric}")
            plt.plot(self.val_loss, label=f"Validation {metric}")
        elif metric == "accuracy":
            plt.ylim(0, 0.8)
            plt.plot(self.train_acc, label=f"Train {metric}")
            plt.plot(self.val_acc, label=f"Validation {metric}")
        else:
            plt.ylim(0, 4)
            plt.plot(self.train_cost, label=f"Train {metric}")
            plt.plot(self.val_cost, label=f"Validation {metric}")
        
        plt.xlabel("epochs")
        plt.ylabel(metric)
        plt.title(f"Monitoring of {metric} during {epochs} epochs.")
        plt.legend()
        plt.savefig(
            f'History/{experiment}_hist_{metric}_{epochs}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.png')
        plt.show()

    def majorityVoting(self, X, y, GDparams, n_cycle=3, experiment="ensemble_learning"):
        epochs, batch_size, eta_min, eta_max, ns = GDparams["n_epochs"], GDparams[
            "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"]
        predictions = []
        for c in range(n_cycle):
            layers = np.load(
                f'History/{experiment}_layers_{c}_{epochs}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.npy', allow_pickle=True)
            model = MLP()
            model.layers = layers
            P = model.forwardpass(X)
            predictions.append(np.argmax(P, axis=0))
        predictions = np.array(predictions)
        majority_voting_class = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[1])]
        return majority_voting_class, accuracy_score(y, majority_voting_class)
        
    def estimateBoundaries(self, X, Y, y, X_val, Y_val, y_val, eta_min, eta_max, n_search, h, lamda, seed=42):
        accuracies = []
        etas = np.linspace(eta_min, eta_max, n_search)
        for eta in etas:
            GDparams = {"n_batch": 100, "n_epochs": 3, "eta_min": eta, "eta_max":0, "ns":0}
            model = MLP(dims=[3072, h, 10], lamda=lamda, seed=seed)
            model.minibatchGD(X, Y, y, X_val, Y_val, y_val, GDparams, verbose=False)
            accuracies.append(model.val_acc[-1])
        return etas, accuracies

    def plotAccuracies(self, etas, accuracies, lamda, h):
        plt.plot(etas, accuracies)
        plt.xlabel("Learning Rate")
        plt.ylabel("Accuracy")
        plt.title(f'Accuracies vs learning rate - lamda={lamda} h={h}')
        plt.legend()
        plt.savefig(f'History/boundaries_{lamda}_{h}.png')
        plt.show()


