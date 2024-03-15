from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(model, X, actuals):
    """Evaluate model predictions."""
    predictions = model.predict(X)
    accuracy = accuracy_score(actuals, predictions.argmax(axis = 1))
    f1 = f1_score(actuals, predictions.argmax(axis = 1), average = 'weighted')
    conf_matrix = confusion_matrix(actuals, predictions.argmax(axis = 1))
    return accuracy, f1, conf_matrix