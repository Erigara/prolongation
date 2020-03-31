#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:04:44 2020

@author: erigara
"""
import pandas as pd
import joblib
import logging
from io import StringIO

def predict(df, model, featues): 
    '''
    Return predictions for given data.

    Parameters
    ----------
    df : pd.Dataframe
        input dataframe.
    model : 
        object with predict method: accept dataframe 
        as input and produce array of class labels.
    featues : list of columns
        subset of df's columns used for making predictions.

    Returns
    -------
    pred : numpy.ndarray
        array of class labels '0' or '1'.

    '''
    x = df.loc[:, featues]
    pred = model.predict(x)
    return pred

def predict_prob(df, model, features):
    '''
    Return predictions probabilities for given data.

    Parameters
    ----------
    df : pd.Dataframe
        input dataframe.
    model :
        object with predict_proba method: accept dataframe 
        as input and produce array of class probabilities between [0, 1].
    features : list of columns
        subset of df's columns used for making predictions.

    Returns
    -------
    pred : numpy.ndarray
        rray of class probabilities between [0, 1].

    '''
    x = df.loc[:, features]
    pred = model.predict_proba(x)[:, 1]
    return pred

# mapping dict that take data type as input 
# and return method to convert that type to dataframes
supported_raw_data_types = {
    'text/csv' : lambda raw_data : pd.read_csv(StringIO(raw_data), sep=';'),
    'application/json' : lambda raw_data : pd.read_json(StringIO(raw_data))
    }

def to_dataframe(raw_data, data_type, column_types):
    '''
    Create dataframe from input raw data in data_type format.

    Parameters
    ----------
    raw_data : str
        input data.
    data_type : str
        mime type of input data.
    column_types : dict
        dict that contain information about dataframe's columns types.

    Returns
    -------
    data : pd.Dataframe
        dataframe created from input data.

    '''
    data = None
    if data_type in supported_raw_data_types:
        try:
            data = supported_raw_data_types[data_type](raw_data)
            data = (data.astype(column_types)
                        .set_index('POLICY_ID', drop=True))
        except Exception as e:
            logging.error(e)
            logging.error('Recieve invalid file')
    else:
        logging.error('Recieve file in unsupported format')
   
    return data

# mapping dict that take data type as input 
# and return method to convert dataframe to that type
supported_return_data_types = {
    'text/csv' : lambda data: data.to_csv(sep=';'),
    'application/json': lambda data: data.reset_index().to_json()
    }

def to_original_format(data, data_type):
    '''
    Create string in data_type format from dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        input data.
    data_type : str
        mime type of output string.

    Returns
    -------
    data : str
        string in data_type format.

    '''
    if data_type in supported_return_data_types:
        try:
            data = supported_return_data_types[data_type](data)
        except Exception as e:
            logging.error(e)
    else:
        logging.error('Transformation to unsupproted format')
    return data

def prediction_pipeline(raw_data, data_type, modeldatapath):
    '''
    Create prediction from input raw_data 
    and return them as string in data_type format.

    Parameters
    ----------
    raw_data : str
        input data.
    data_type : str
        mime type of input data.
    modeldatapath : pathlike
        path to model data used for making predictions.

    Returns
    -------
    predictions : str
        predictions in data_type format.

    '''
    model_data = joblib.load(modeldatapath)
    model = model_data['model']
    types = model_data['types']
    features = model_data['features']
    
    data = to_dataframe(raw_data, data_type, types)
    if not data is None:
        predictions = (data.assign(POLICY_IS_RENEWED=lambda df: predict(df, model, features),
                                   POLICY_IS_RENEWED_PROBABILITY=lambda df: predict_prob(df, model, features))
                           .loc[:, ['POLICY_IS_RENEWED', 'POLICY_IS_RENEWED_PROBABILITY']])
        predictions = to_original_format(predictions, data_type)
        
        return predictions