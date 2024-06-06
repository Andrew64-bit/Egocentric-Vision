import numpy as np
from scipy.stats import kurtosis, skew
from statsmodels.tsa.ar_model import AutoReg

def EMG_Feature_Extractor(segment):
    features = []
    # Integrated EMG (IEMG)
    IEMG = np.sum(np.abs(segment), axis=0)
    features.append(IEMG)
    
    # Mean Squared Value (MSV)
    MSV = np.mean(segment**2, axis=0)
    features.append(MSV)
    
    # Variance
    VAR = np.var(segment, axis=0)
    features.append(VAR)
    
    # Root Mean Square (RMS)
    RMS = np.sqrt(MSV)
    features.append(RMS)
    
    # ln RMS
    ln_RMS = np.log(RMS + 1e-8)  # Aggiungi un piccolo valore per evitare log(0)
    features.append(ln_RMS)
    
    # Kurtosis
    KURT = kurtosis(segment, axis=0)
    features.append(KURT)
    
    # Skewness
    SKEW = skew(segment, axis=0)
    features.append(SKEW)
    
    # Auto-regressive Model (ARM)
    ARM_features = []
    for ch in range(segment.shape[1]):
        model = AutoReg(segment[:, ch], lags=4, old_names=False).fit()
        ARM_features.extend(model.params)
    features.append(ARM_features)
    
    # Concatenare tutte le feature in un singolo vettore
    feature_vector = np.concatenate(features)
    return feature_vector