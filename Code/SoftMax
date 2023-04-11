class SoftMax: 
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs), axis= 1, keepdims= True)   #inputs = inputs - maxinput   to prevent the overflow to inf
        probalities = exp_values / np.sum(exp_values, axis= 1, keepdims= True)
        self.output = probalities