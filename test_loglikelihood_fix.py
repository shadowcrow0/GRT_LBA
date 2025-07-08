#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_loglikelihood_fix.py - 測試 loglikelihood 修復是否有效
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """測試所有必要的模組是否可以導入"""
    print("📦 測試模組導入...")
    
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError:
        print("  ❌ numpy - 請安裝: pip install numpy")
        return False
    
    try:
        import pymc as pm
        print("  ✓ pymc")
    except ImportError:
        print("  ❌ pymc - 請安裝: pip install pymc")
        return False
    
    try:
        import arviz as az
        print("  ✓ arviz")
    except ImportError:
        print("  ❌ arviz - 請安裝: pip install arviz")
        return False
    
    try:
        import pytensor
        print("  ✓ pytensor")
    except ImportError:
        print("  ❌ pytensor - 請安裝: pip install pytensor")
        return False
    
    return True

def test_data_files():
    """測試數據檔案是否存在"""
    print("\n📁 檢查數據檔案...")
    
    files_to_check = [
        'model_data.npz',
        'model_data_fixed.npz'
    ]
    
    available_files = []
    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✓ {file}")
            available_files.append(file)
        else:
            print(f"  ❌ {file}")
    
    if not available_files:
        print("  ⚠️ 沒有找到數據檔案，創建測試數據...")
        return create_test_data()
    
    return available_files[0]

def create_test_data():
    """創建測試數據"""
    try:
        import numpy as np
        
        print("  🔧 創建測試數據...")
        
        n_trials = 100
        test_data = np.zeros((n_trials, 2))
        test_data[:, 0] = np.random.uniform(300, 1500, n_trials)  # RT in ms
        test_data[:, 1] = np.random.binomial(1, 0.8, n_trials)   # Accuracy
        
        participant_idx = np.zeros(n_trials, dtype=int)
        model_input_data = {
            'left_match': np.random.binomial(1, 0.5, n_trials).astype(float),
            'right_match': np.random.binomial(1, 0.5, n_trials).astype(float)
        }
        
        np.savez('test_data_for_fix.npz',
                 observed_value=test_data,
                 participant_idx=participant_idx,
                 model_input_data=model_input_data)
        
        print("  ✓ 測試數據已創建: test_data_for_fix.npz")
        return 'test_data_for_fix.npz'
        
    except Exception as e:
        print(f"  ❌ 創建測試數據失敗: {e}")
        return None

def test_fix_functions():
    """測試修復函數"""
    print("\n🔧 測試修復函數...")
    
    try:
        from LBA_loglikelihood_fix import (
            diagnose_loglikelihood_issue,
            create_fixed_coactive_model,
            test_fixed_model
        )
        print("  ✓ 修復函數導入成功")
        return True
    except ImportError as e:
        print(f"  ❌ 修復函數導入失敗: {e}")
        print("  請確保 LBA_loglikelihood_fix.py 存在於當前目錄")
        return False
    except Exception as e:
        print(f"  ❌ 修復函數測試失敗: {e}")
        return False

def run_basic_test(data_file):
    """運行基本測試"""
    print(f"\n🧪 運行基本測試 (使用 {data_file})...")
    
    try:
        from LBA_loglikelihood_fix import test_fixed_model
        
        success = test_fixed_model(data_file)
        
        if success:
            print("  ✅ 基本測試通過")
            return True
        else:
            print("  ❌ 基本測試失敗")
            return False
            
    except Exception as e:
        print(f"  ❌ 基本測試出錯: {e}")
        return False

def run_main_script_test():
    """測試主腳本是否可以運行"""
    print("\n🚀 測試修復版主腳本...")
    
    if not os.path.exists('LBA_main_fixed.py'):
        print("  ❌ LBA_main_fixed.py 不存在")
        return False
    
    try:
        # 測試導入
        import LBA_main_fixed
        print("  ✓ LBA_main_fixed.py 導入成功")
        
        # 測試創建分析器
        runner = LBA_main_fixed.FixedLBAAnalysisRunner()
        print("  ✓ 修復版分析器創建成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 主腳本測試失敗: {e}")
        return False

def show_usage_instructions():
    """顯示使用說明"""
    instructions = """
    
    📋 使用說明:
    
    1. 基本測試 (推薦先執行):
       python test_loglikelihood_fix.py
    
    2. 運行修復版分析:
       python LBA_main_fixed.py --test
    
    3. 分析特定參與者:
       python LBA_main_fixed.py --participant 1
    
    4. 使用特定數據檔案:
       python LBA_main_fixed.py --data-file model_data_fixed.npz --test
    
    5. 如果原來的 LBA_main.py 仍然有問題，可以完全替換:
       - 備份原檔案: cp LBA_main.py LBA_main_backup.py
       - 使用修復版: cp LBA_main_fixed.py LBA_main.py
    
    🔧 故障排除:
    
    - 如果出現 "Bad initial energy" 錯誤:
      * 檢查數據單位是否正確 (RT 應該是毫秒)
      * 嘗試使用 model_data_fixed.npz
    
    - 如果採樣失敗:
      * 減少 draws 和 tune 參數
      * 增加 max_attempts
    
    - 如果 WAIC 計算失敗:
      * 檢查 log_likelihood 是否存在
      * 使用備用比較方法
    
    """
    
    print(instructions)

def main():
    """主測試函數"""
    
    print("🔍 LBA Log-likelihood 修復測試")
    print("=" * 50)
    
    # 1. 測試模組導入
    if not test_imports():
        print("\n❌ 模組導入失敗，請先安裝必要的套件")
        print("pip install pymc arviz numpy pandas")
        return False
    
    # 2. 檢查數據檔案
    data_file = test_data_files()
    if not data_file:
        print("\n❌ 無法找到或創建數據檔案")
        return False
    
    # 3. 測試修復函數
    if not test_fix_functions():
        print("\n❌ 修復函數測試失敗")
        return False
    
    # 4. 運行基本測試
    if not run_basic_test(data_file):
        print("\n❌ 基本測試失敗")
        print("這可能是由於數據問題或模型設置問題")
        return False
    
    # 5. 測試主腳本
    if not run_main_script_test():
        print("\n❌ 主腳本測試失敗")
        return False
    
    print("\n✅ 所有測試通過！")
    print("\n🎉 log-likelihood 修復已準備就緒")
    
    # 顯示使用說明
    show_usage_instructions()
    
    return True

if __name__ == '__main__':
    success = main()
    
    if success:
        print("\n🚀 你現在可以使用修復版程式了:")
        print("python LBA_main_fixed.py --test")
    else:
        print("\n❌ 測試失敗，請檢查錯誤訊息並修復問題")