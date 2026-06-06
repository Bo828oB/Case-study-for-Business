"""
Data Science Case Study - data selection module
Author: BO
Data: 2026-06-03
Description: This module provides some funtions for feature selection,  VIF

"""

import os
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_bool_dtype, is_numeric_dtype
from dython.nominal import associations
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def auto_vif_drop(X_matrix, thresh=5.0):
    """
    preprocessed features X
    threshold: normally VIF <= 5 shows lower or no correlation

    return: cleaned features X


    """
    
    X_current = X_matrix.copy()
    

   
    while True:
        
        X_with_const = sm.add_constant(X_current)
        
       
        try:
            vifs = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
        except Exception as e:
            print(f"reverse matrix error: {e}")
            
            X_with_const = X_with_const + np.random.normal(0, 1e-9, X_with_const.shape)
            vifs = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
            
        vif_df = pd.DataFrame({'feature': X_with_const.columns, 'VIF': vifs})
        vif_df = vif_df[vif_df['feature'] != 'const'] 
        
        
        max_vif = vif_df['VIF'].max()
        max_feature = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
        
        
        if max_vif < thresh or np.isnan(max_vif):
            break
            
        
        print(f"remove [{max_feature}] because of the highest VIF value")
        X_current = X_current.drop(columns=[max_feature])
        
    return X_current


def select_features_by_rf(X, y, threshold, max_depth, n_estimators, random_state):
    """
    Random Forest to select the feautes 
    return:
    - selected_features: list,
    - importance_df: pd.DataFrame,
    """
    print(f"Beginning: original features {X.shape[1]} )")
    # 1. Build Random Model
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state, 
        n_jobs=-1
    )
    rf_model.fit(X, y)
    
    # 2. select the importance 
    importance_df = pd.DataFrame({
        'Feature': X.columns, 
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
    passed_mask = importance_df['Cumulative_Importance'] <= threshold
    
    # error dealing
    if not passed_mask.any():
        selected_features = [importance_df['Feature'].iloc[0]]
    else:
        
        selected_features = importance_df[passed_mask]['Feature'].tolist()
        
        

    print(f"finished:original features {X.shape[1]} 个 -> selected features: {len(selected_features)} ,cumulative importance {threshold})")
    
    return selected_features, importance_df
