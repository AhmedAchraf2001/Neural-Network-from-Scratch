class categoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 10e-7, 1 - 10e-7)
        if (y_true.shape[0]) == 1 : #user sent scaller values of classes
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif (y_true.shape[0]) == 2 :  #user sent one hot encoded matrix for classes
            correct_confidences = np.sum(y_pred_clipped*y_true , axis= 1 )
        neg_log = - np.log(correct_confidences)
        return neg_log