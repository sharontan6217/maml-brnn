import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class selfTrainingClassifier():
    def selfTrainingClassifier(model_predict,x_train,y_train_mask,x_test,y_test):
        masked_index = [i for i, j in enumerate(y_train_mask) if j==-1]
        for idx in masked_index:
            x_train.pop(idx)


        return None
    def LabelSpreading(model_predict,x_train,y_train,x_test,y_test):
        split_labelled = int(len(x_train)*0.8)
        X_labeled, y_labeled = x_train[:split_labelled], y_train[:split_labelled]
        X_unlabeled, y_unlabeled = x_train[split_labelled:], y_train[split_labelled:]
        model_predict.fit(X_labeled, y_labeled, validation_data=[x_test,y_test])

        confidence_threshold = 0.9  
        for iteration in range(5):  # Run 5 iterations
            pseudo_labels = model_predict.predict(X_unlabeled)  # Generate pseudo-labels
            pseudo_probabilities = model_predict.predict_proba(X_unlabeled).max(axis=1)  # Get confidence scores
            
            confidence_threshold = 0.9  # Confidence threshold for pseudo-labeling
            confident_indices = np.where(pseudo_probabilities > confidence_threshold)[0]  # Identify confident samples

            # Add confident pseudo-labeled samples to the labeled dataset
            X_labeled = np.vstack((X_labeled, X_unlabeled[confident_indices]))
            y_labeled = np.hstack((y_labeled, pseudo_labels[confident_indices]))

            # Remove pseudo-labeled samples from the unlabeled set
            X_unlabeled = np.delete(X_unlabeled, confident_indices, axis=0)

            # Retrain the model with the expanded labeled dataset
            model_predict.fit(X_labeled, y_labeled)

        # Predict labels on the test dataset
        y_predict = model_predict.predict(x_test)

        # Print accuracy
        print("Final Model Accuracy on Test Data:", accuracy_score(y_test, y_pred))
        return y_predict,model_predict