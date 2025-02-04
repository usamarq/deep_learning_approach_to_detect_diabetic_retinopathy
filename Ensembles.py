# Defining Ensemble Methods here to use in part (d)

import numpy as np
import pandas as pd
import os

# Ensemble method: Max Voting
def max_voting(*predictions):
    return np.array([np.bincount(pred).argmax() for pred in zip(*predictions)])

def weighted_average(predictions, weights):
    predictions = np.array(predictions)
    weights = np.array(weights)
    weighted_preds = np.average(predictions, axis=0, weights=weights)
    
    return np.round(weighted_preds).astype(int)



# for Kaggle Predictions file
def preds_to_csv(pred_path, ids, preds): 

    ensemble_df = pd.DataFrame({
        'ID': ids, 
        'TARGET': preds
    })
    ensemble_df.to_csv(pred_path, index=False)
    print(f'Saved predictions to {os.path.abspath(pred_path)}')


