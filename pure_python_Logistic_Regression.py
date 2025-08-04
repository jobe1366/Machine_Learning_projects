#! usr/bin/python3


class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.
    """
    def __init__(self, learning_rate=0.1, n_iter=1500, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.learning_rate * X.T.dot(errors) / X.shape[0]
            self.b_ += self.learning_rate * errors.mean()
            loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X.shape[0]
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
       
    def LR_accuracy_score(self , y_prediction , y_test):
        
     score= [ 1  if pred==tst else 0     for pred, tst in zip( y_prediction, y_test)  ]
     return  np.round(np.sum(score)/len(y_test)  , 3)
 
 
    def predict_proba(self, X):
        
        net_input = self.net_input(X)
        probability_positive_class = self.activation(net_input)
        probability_negative_class = 1 - probability_positive_class
        return np.column_stack((probability_negative_class, probability_positive_class))
    
    
    
    
    
if __name__ == "__main__":
    
    
        from sklearn.datasets   import  load_breast_cancer
        from sklearn.model_selection import  train_test_split
        from sklearn.preprocessing  import StandardScaler
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        
        
        cancer = load_breast_cancer()
        X_train , X_test ,Y_train , Y_test = train_test_split(cancer.data[:,[26]], cancer.target, test_size= 0.25  )

        sc = StandardScaler()
        sc.fit(X_train)   
        X_train_std = sc.transform(X_train) 
        
    
        LR_model = LogisticRegressionGD( learning_rate=0.1, n_iter=1000, random_state=1)  
        LR_model.fit(X_train, Y_train)
        
        
        print("\n-----------------------\n")
        probabilities = LR_model.predict_proba(X_test[:5 , :])
        print("Predicted Probabilities:\n", probabilities)
        
        print("\n-----------------------\n")
        test_predictions = LR_model.predict(X_test)
        print("\nPredicted Classes:\n", test_predictions)
        
        train_predictions = LR_model.predict(X_train_std)
        
        print("\n-----------------------\n")
        accuracy_on_testdata = LR_model.LR_accuracy_score(test_predictions  , Y_test)
        print(f"accuracy_on_testdata is: \n {accuracy_on_testdata}")
        
        
        print("\n-----------------------\n")
        accuracy_on_traindata = LR_model.LR_accuracy_score(train_predictions , Y_train)
        print(f"accuracy_on_traindata is: \n {accuracy_on_traindata}")
        
        print("\n-----------------------\n")
        
        confmat = confusion_matrix(y_true=Y_test, y_pred=test_predictions)
        print('Confusion matrix is: \n',confmat)
            