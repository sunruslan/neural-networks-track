import numpy as np
## Layes
class Linear:
    def __init__(self, input_size, output_size, no_b=False):
        '''
        Creates weights and biases for linear layer from N(0, 0.01).
        Dimention of inputs is *input_size*, of output: *output_size*.
        no_b=True - do not use interception in prediction and backward (y = w*X)
        '''
        self.no_b = no_b
        self.b = np.random.normal(loc=0, scale = 0.01, size=(1, output_size))
        self.W = np.random.normal(loc=0, scale=0.01, size=(input_size, output_size))

    # N - batch_size
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, input_size).
        Returns output of size (N, output_size).
        Hint: You may need to store X for backward pass
        '''
        self.X = np.copy(X)
        result = np.matmul(self.X, self.W)
        if not self.no_b:
            result += np.matmul(np.ones((X.shape[0], 1)), self.b) 
        return result

    def backward(self, dLdy):
        '''
        1. Compute dLdw and dLdx.
        2. Store dLdw for step() call
        3. Return dLdx
        '''
        if not self.no_b:
            self.dLdb = np.matmul(np.ones((1, self.X.shape[0])), dLdy)
        self.dLdw = np.matmul(dLdy.T, self.X).T
        self.dLdx = np.matmul(dLdy, self.W.T)
        return self.dLdx

    def step(self, learning_rate):
        '''
        1. Apply gradient dLdw to network:
        w <- w - l*dLdw
        '''
        self.b = self.b-learning_rate*self.dLdb
        self.W = self.W-learning_rate*self.dLdw


## Activations
class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self, X):
        '''
        Passes objects through this layer.
        X is np.array of size (N, d)
        '''
        self.sigmoid = 1./(1+np.exp(-X))
        return self.sigmoid

    def backward(self, dLdy):
        '''
        1. Compute dLdx.
        2. Return dLdx
        '''
        return dLdy*(1-self.sigmoid)*self.sigmoid

    def step(self, learning_rate):
        pass

class ELU:
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, X):
        self.X = np.copy(X)
        return self.alpha*(np.exp(X)-1)*(X < 0)+X*(X >= 0)

    def backward(self, dLdy):
        return dLdy*(self.alpha*np.exp(self.X)*(self.X < 0)+(self.X >= 0))

    def step(self, learning_rate):
        pass


class ReLU:
    def __init__(self, a):
        self.a = a

    def forward(self, X):
        self.X = np.copy(X)
        return self.a*X*(X < 0)+X*(X >= 0)
      
    def backward(self, dLdy):
        return dLdy*(self.a*(self.X < 0)+(self.X >= 0))

    def step(self, learning_rate):
        pass


class Tanh:
    def forward(self, X):
        self.tanh = np.tanh(X)
        return self.tanh

    def backward(self, dLdy):
        return dLdy*(1-self.tanh**2)

    def step(self, learning_rate):
        pass


## Final layers, loss functions
class SoftMax_NLLLoss:
    def __init__(self):
        '''
        Applies Softmax operation to inputs and computes NLL loss
        '''
        pass

    def forward(self, X):
        '''
        Returns SoftMax for all X (matrix with size X.shape, containing in lines probabilities of each class)
        '''
        self.X = np.copy(X)
        self.smax = np.exp(X)
        self.smax /= self.smax.sum(1, keepdims=True)
        return self.smax

    # y - true labels. Calculates dL/dy, returns dL/dX
    def backward(self, y):
        self.y = np.zeros(self.X.shape)
        self.y[np.arange(self.X.shape[0]), y] = 1
        return (self.smax - self.y) / self.y.shape[0]
    
    def get_loss(self, y):
        res = 0
        for i in range(self.smax.shape[0]):
            res -= np.log(self.smax[i][y[i]])
        return res/self.smax.shape[0]
                


class MSE_Error:
    # Saves X for backprop, X.shape = N x 1
    def forward(self, X):
        self.X = np.copy(X)
        return X

    # Returns dL/dy (y - true labels)
    def backward(self, y):
        y.resize(self.X.shape)
        return 2 * (self.X - y) / self.X.shape[0]
    
    def get_loss(self, y):
        y.resize(self.X.shape)
        return np.sum((self.X - y) ** 2) / self.X.shape[0]


## Main class
# loss_function can be None - if the last layer is SoftMax_NLLLoss: it can produce dL/dy by itself
# Or, for example, loss_function can be MSE_Error()
class NeuralNetwork:
    def __init__(self, modules, loss_function=None):
        '''
        Constructs network with *modules* as its layers
        '''
        self.modules = modules
        self.loss_function = loss_function
    
    def forward(self, X):
        for module in self.modules:
            X = module.forward(X)
        return X

    def get_loss(self, y):
        return self.modules[-1].get_loss(y)
    
    # y - true labels.
    # Calls backward() for each layer. dL/dy from k+1 layer should be passed to layer k
    # First dL/dy may be calculated directly in last layer (if loss_function=None) or by loss_function(y)
    def backward(self, y):
        dLdy = self.modules[-1].backward(y) if self.loss_function is None else loss_function.backward(y)
        for module in self.modules[-2::-1]:
            dLdy = module.backward(dLdy)
        return dLdy

    # calls step() for each layer
    def step(self, learning_rate):
        for module in self.modules[:-1]:
            module.step(learning_rate)