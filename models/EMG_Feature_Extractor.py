import numpy as np
from scipy.stats import kurtosis, skew
from statsmodels.tsa.ar_model import AutoReg
import torch

def EMG_Feature_Extractor(segment):
    features = []
    # Integrated EMG (IEMG)
    print(f"Segment Shape: {segment.shape}")    
    IEMG = torch.sum(torch.abs(segment), dim=0).to(segment.device)
    print(f"IEMG: {IEMG.shape}")
    features.append(IEMG)
    
    # Mean Squared Value (MSV)
    MSV = torch.mean(segment**2, dim=0).to(segment.device)
    print(f"MSV: {MSV.shape}")
    features.append(MSV)

    # Variance
    VAR = torch.var(segment, dim=0).to(segment.device)
    print(f"VAR: {VAR.shape}")
    features.append(VAR)

    # Root Mean Square (RMS)
    RMS = torch.sqrt(MSV).to(segment.device)
    features.append(RMS)

    # ln RMS
    ln_RMS = torch.log(RMS + 1e-8).to(segment.device)  # Add a small value to avoid log(0)
    features.append(ln_RMS)

    # Kurtosis and Skewness (using scipy, so convert to numpy and back)
    segment_np = segment.cpu().numpy()  # Ensure it's on CPU for scipy
    KURT = torch.tensor(kurtosis(segment_np, axis=0)).to(segment.device)
    features.append(KURT)

    SKEW = torch.tensor(skew(segment_np, axis=0)).to(segment.device)
    features.append(SKEW)


    # Concatenate all features into a single vector
    feature_vector = torch.cat(features, dim=0)
    
    return feature_vector