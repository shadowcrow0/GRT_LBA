"""
生產級 LBA 模型 - 基於成功的調試結果
優化了數值穩定性和採樣效率
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

def stable_lba_loglik(rt_data, choice_data, participant_idx, A, b, v1, v2, t0):
    """
    數值穩定的 LBA 對數似然函數
    支持層次模型結構
    """
    # 參數安全性檢查
    A = pt.maximum(A, 0.05)
    b = pt.maximum(b, A + 0.05)
    v1 = pt.maximum(v1, 0.1)
    v2 = pt.maximum(v2, 0.1)
    t0 = pt.maximum(t0, 0.01)
    
    # 計算決策時間
    rt_decision = pt.maximum(rt_data - t0[participant_idx], 0.01)
    
    # 向量化計算以提高效率
    n_trials = rt_data.shape[0]
    loglik_trials = pt.zeros(n_trials)
    
    for i in range(n_trials):
        p_idx = participant_idx[i]
        choice_i = choice_data[i]
        rt_i = rt_decision[i]
        
        # 當前參與者的參數
        A_i = A[p_idx]
        b_i = b[p_idx]
