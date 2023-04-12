class Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1+ np.exp(-1*inputs))