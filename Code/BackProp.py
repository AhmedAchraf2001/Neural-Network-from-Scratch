class optimization:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.baises = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = inputs@self.weights + self.baises
        
    def backprop(self, y_pred, y_true):
        new_b = [np.zeros(b.shape) for b in self.baises]
        new_w = [np.zeros(w.shape) for w in self.weights]
        zs = self.forward(inputs)
        activations = Sigmoid(zs)
        
         # backward propagation
        delta = self.Lossfunction(activations[-1], y) * Sigmoid_dervatied(zs[-1])
        new_b[-1] = delta
        new_w[-1] = delta@activations[-2].T
        for l in xrange(2, self.n_inputs):
            z = zs[-l]
            sp = Sigmoid_dervatied(z)
            delta = self.weights[-l + 1].T@delta * sp
            new_b[-l] = delta
            new_w[-l] = delta@activations[-l-1].T
        return (new_b, new_w)
    
    def update_mini_batch(self, mini_batch, eta):
        vector_b = [np.zeros(b.shape) for b in self.baises]
        vector_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_vector_b, delta_vector_w = self.backprop(x, y)
            vector_b = [nb + dnb for nb, dnb in zip(vector_b, delta_vector_b)]
            vector_w = [ nw + dnw for nw, dnw in zip(vector_w, delta_vector_w)]
        self.weights = [ w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, vector_w)]
        self.biases = [ b - (eta/len(mini_batch))*nb for b, nb in zip(self.baises, vector_b)]
        
    def Lossfunction(self, y_pred, y_true):
        return y_pred - y_true
    
    def Sigmoid(self, inputs):
        return 1/(1 + np.exp(-1 * inputs))
    
    def Sigmoid_dervatied(self, inputs):
        return optimization.Sigmoid(inputs)*( 1 - optimization.Sigmoid(inputs))