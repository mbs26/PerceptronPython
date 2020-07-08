import numpy as np

Sigmoid = lambda x: 1.0/(1.0 + np.exp(-x))
dSigmoid = lambda x: x * (1.0 - x)

Error = lambda y,ex: np.mean((y-ex)**2)
dError = lambda y,ex: y-ex

class RN:
    def __init__(self, estructura):
        self.estructura = estructura

        self.w = [] # w[0] conecta capa -1 (entrada) con capa 0
        self.u = [] # u[0] pesos de capa 0
        self.n = []

        for c in range(1, len(self.estructura)):
            self.w.append(np.random.randn(self.estructura[c - 1], self.estructura[c]))
            self.u.append(np.random.randn(1,self.estructura[c]))
            self.n.append(np.zeros(self.estructura[c]))

    def f_prop(self, data):
        self.n[0] = Sigmoid(data.dot(self.w[0]) + self.u[0])

        for c in range(1, len(self.estructura) - 1):
            self.n[c] = Sigmoid(self.n[c - 1].dot(self.w[c]) + self.u[c])

        return self.n[-1]

    def train(self, data_set, expected_set, alpha = 0.1 , t=True):
        # forard_prop
        neuronas = [data_set]
        for l in range(len(self.estructura) - 1 ):
            neuronas.append(Sigmoid(neuronas[-1].dot(self.w[l]) + self.u[l]))

        if t:
            # back_prop
            deltas = []
            for l in reversed(range(len(self.estructura) - 1)):
                if l == len(self.estructura) - 2:
                    deltas.insert(0, dError(neuronas[-1], expected_set) * dSigmoid(neuronas[l+1]))
                else:
                    deltas.insert(0, deltas[0].dot(_W) * dSigmoid(neuronas[l + 1]))

                _W = self.w[l].T

                # actualizar_valores
                self.u[l] -= alpha * np.sum(deltas[0], axis=0, keepdims=True)
                self.w[l] -= alpha * neuronas[l].T.dot(deltas[0])

        return Error(neuronas[-1], expected_set)

def ejemplo_circulos():
    from sklearn.datasets import make_circles
    import matplotlib.pyplot as plt

    red = RN([2,4,1])

    X, Y = make_circles(500, factor = 0.5, noise = 0.05)
    Y=Y.reshape((500,1))

    error = []
    for i in range(3000):
      e = red.train(X,Y, alpha = 0.01)

      if i%1000==0:
        error.append(e)

        _x0= np.linspace(-1.5,1.5, 50)
        _x1= np.linspace(-1.5,1.5, 50)

        _Y= np.zeros((50,50))

        for i0,x0 in enumerate(_x0):
          for i1,x1 in enumerate(_x1):
            _Y[i0,i1] = red.f_prop(np.array([x0,x1]))[0]

        plt.pcolormesh(_x0,_x1,_Y, cmap='coolwarm')
        plt.axis('equal')

        plt.scatter(X[Y[:,0]==0,0],X[Y[:,0]==0,1], c='skyblue')
        plt.scatter(X[Y[:,0]==1,0],X[Y[:,0]==1,1], c='salmon')

        plt.show()
        plt.plot(range(len(error)),error)
        plt.show()
        
def ejemplo_mnist():
    from tensorflow.keras.datasets import mnist

    (X_train, Y2_train), (X_test, Y2_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)/255
    X_test = X_test.reshape(10000, 784)/255

    Y_train = np.zeros((60000,10))
    for i in range(60000):
        Y_train[i][Y2_train[i]]=1

    Y_test = np.zeros((10000,10))
    for i in range(10000):
        Y_test[i][Y2_test[i]]=1

    red = RN([784,16,16,10])

    for i in range(10):
        for k in range(120):
            red.train(X_train[500*k:500*(k+1)],Y_train[500*k:500*(k+1)], alpha = 0.01)

        print(red.train(X_test, Y_test, t=False))
        print(red.f_prop(X_test[0]))


def ejemplo_mnist2():
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    X_test = np.load('X_test.npy')
    Y_test = np.load('Y_test.npy')

    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    red = RN([784,16,16,10])

    for i in range(10):
        for k in range(120):
            red.train(X_train[500*k:500*(k+1)],Y_train[500*k:500*(k+1)], alpha = 0.01)

        print(red.train(X_test, Y_test, t=False))
        print(red.f_prop(X_test[0]))

ejemplo_mnist()
