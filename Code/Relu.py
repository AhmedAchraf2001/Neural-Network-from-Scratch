class Relu:
    def forward(self, input):
        self.output = np.maximum(0, input)
