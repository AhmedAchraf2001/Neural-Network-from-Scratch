class Accuracy:
    def forward(self, outputs, class_target):
        predications = np.argmax(outputs, axis= 1)
        accuracy = np.mean(predications == class_target)
        return accuracy