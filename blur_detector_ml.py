import pandas as pd
from pycaret.classification import *

# Loading the saved model
model = load_model('tuned_gbm_hyperopt_05032021')

def ml_predict(FFT,LaplaceVar,LaplaceVarGrid):
    data_unseen = pd.DataFrame([(FFT,LaplaceVar,LaplaceVarGrid)], columns = ['FFT','Laplace_Var','Laplace_Var_Grid'])
    new_prediction = predict_model(model, data=data_unseen)
    return new_prediction['Label'][0]