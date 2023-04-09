class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.baises = np.zeros((1, n_neurons))
        
    def forword(self, inputs):
        self.output = inputs@self.weights + self.baises
