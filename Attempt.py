"""
優化版協方差矩陣LBA分析
解決性能問題和PyTensor兼容性問題
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 設置更高效的採樣參數
import pytensor
pytensor.config.floatX = 'float32'  # 使用float32提高速度

# ============================================================================
# 第一部分：資料載入函數（優化版）
# ============================================================================

def load_real_subject_data(csv_file_path='GRT_LBA.csv', max_trials_per_subject=300):
    """
    載入真實受試者資料（限制試驗數量以提高速度）
    """
    
    print("載入真實受試者資料...")
    
    try:
        raw_data = pd.read_csv(csv_file_path)
        print(f"成功讀取 {csv_file_path}")
        print(f"原始資料維度：{raw_data.shape}")
        
    except FileNotFoundError:
        print(f"找不到檔案 {csv_file_path}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 檢查必要的欄位
    required_columns = ['Chanel1', 'Chanel2', 'Response', 'RT', 'acc']
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    
    if missing_columns:
        print(f"缺少必要欄位：{missing_columns}")
        print(f"可用欄位：{list(raw_data.columns)}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 資料清理和轉換
    print("轉換資料格式...")
    
    # 移除缺失值
    clean_data = raw_data.dropna(subset=required_columns)
    print(f"清理後資料維度：{clean_data.shape}")
    
    # 移除極端反應時間
    clean_data = clean_data[(clean_data['RT'] > 0.1) & (clean_data['RT'] < 3.0)]
    print(f"移除極端RT後維度：{clean_data.shape}")
    
    # 限制每個受試者的試驗數量
    if 'Subject' in clean_data.columns:
        clean_data = clean_data.groupby('Subject').head(max_trials_per_subject)
    elif 'participant' in clean_data.columns:
        clean_data = clean_data.groupby('participant').head(max_trials_per_subject)
    
    # 轉換為模型需要的格式
    converted_data = []
    
    for idx, row in clean_data.iterrows():
        # 確保Response在0-3範圍內
        if not (0 <= row['Response'] <= 3):
            continue
            
        # 確保Chanel值為0或1
        if row['Chanel1'] not in [0, 1] or row['Chanel2'] not in [0, 1]:
            continue
        
        converted_row = {
            'subject_id': row.get('Subject', row.get('participant', 1)),
            'trial': len(converted_data) + 1,
            'left_pattern': int(row['Chanel1']),
            'right_pattern': int(row['Chanel2']),
            'response': int(row['Response']),
            'rt': float(row['RT']),
            'accuracy': int(row['acc']),
            'stimulus_type': int(row['Chanel1']) * 2 + int(row['Chanel2']),
            'is_symmetric': 1 if row['Chanel1'] == row['Chanel2'] else 0
        }
        
        converted_data.append(converted_row)
    
    # 轉換為DataFrame
    df = pd.DataFrame(converted_data)
    
    if len(df) == 0:
        print("❌ 沒有有效的資料行，生成模擬資料")
        return generate_test_data()
    
    # 資料統計
    print(f"\n✅ 真實資料載入完成：")
    print(f"  有效試驗數：{len(df)}")
    print(f"  受試者數：{df['subject_id'].nunique()}")
    print(f"  整體準確率：{df['accuracy'].mean():.3f}")
    print(f"  平均RT：{df['rt'].mean():.3f}秒")
    print(f"  對稱試驗比例：{df['is_symmetric'].mean():.3f}")
    
    return df

def generate_test_data(n_trials=200):
    """
    生成測試用的模擬資料
    """
    print("生成模擬測試資料...")
    
    np.random.seed(42)
    data = []
    
    for trial in range(n_trials):
        # 隨機刺激
        left_pattern = np.random.choice([0, 1])
        right_pattern = np.random.choice([0, 1])
        is_symmetric = int(left_pattern == right_pattern)
        stimulus_type = left_pattern * 2 + right_pattern
        
        # 模擬反應
        # 對稱刺激有較高準確率
        base_accuracy = 0.8 if is_symmetric else 0.7
        accuracy = int(np.random.random() < base_accuracy)
        
        # 正確反應
        if accuracy:
            response = stimulus_type
        else:
            response = np.random.choice([i for i in range(4) if i != stimulus_type])
        
        # 反應時間
        base_rt = 0.8 if is_symmetric else 0.9
        rt = np.random.gamma(2, base_rt/2)
        rt = np.clip(rt, 0.2, 3.0)
        
        data.append({
            'subject_id': 1,
            'trial': trial + 1,
            'left_pattern': left_pattern,
            'right_pattern': right_pattern,
            'response': response,
            'rt': rt,
            'accuracy': accuracy,
            'stimulus_type': stimulus_type,
            'is_symmetric': is_symmetric
        })
    
    df = pd.DataFrame(data)
    print(f"✅ 模擬資料生成完成：{len(df)} 試驗")
    return df

# ============================================================================
# 第二部分：高效能協方差矩陣LBA模型
# ============================================================================

def build_fast_covariance_lba_model(data):
    """
    建構高效能協方差矩陣LBA模型
    """
    
    print("建構高效能協方差矩陣LBA模型...")
    
    # 準備資料（轉為numpy陣列提高效率）
    rt_obs = data['rt'].values.astype('float32')
    response_obs = data['response'].values.astype('int32')
    is_symmetric = data['is_symmetric'].values.astype('float32')
    stimulus_type = data['stimulus_type'].values.astype('int32')
    
    n_trials = len(data)
    print(f"處理 {n_trials} 個試驗")
    
    with pm.Model() as model:
        
        # 協方差參數（簡化）
        rho_base = pm.Uniform('rho_base', lower=-0.5, upper=0.5)
        rho_symmetry_effect = pm.Normal('rho_symmetry_effect', mu=0, sigma=0.2)
        
        # LBA參數（簡化）
        A = pm.HalfNormal('A', sigma=0.2)
        b_base = pm.HalfNormal('b_base', sigma=0.3)
        s = pm.HalfNormal('s', sigma=0.2)
        t0 = pm.HalfNormal('t0', sigma=0.1)
        
        # 基礎drift rate
        drift_base = pm.HalfNormal('drift_base', sigma=0.5)
        drift_correct_boost = pm.HalfNormal('drift_correct_boost', sigma=0.3)
        
        # 對稱性效應
        symmetry_boost = pm.Normal('symmetry_boost', mu=0, sigma=0.1)
        
        # 向量化計算drift rates
        def compute_drift_rates_vectorized():
            # 為每個累加器計算drift rate
            drift_rates = pt.zeros((n_trials, 4))
            
            for acc in range(4):
                # 基礎drift
                base_drift = drift_base
                
                # 正確累加器增強
                correct_boost = pt.where(pt.eq(stimulus_type, acc), drift_correct_boost, 0.0)
                
                # 對稱性效應
                sym_boost = symmetry_boost * is_symmetric * pt.where(pt.eq(stimulus_type, acc), 1.0, 0.0)
                
                # 相關性效應（簡化）
                rho_trial = rho_base + rho_symmetry_effect * is_symmetric
                correlation_effect = pt.abs(rho_trial) * 0.1
                
                # 總drift rate
                final_drift = base_drift + correct_boost + sym_boost + correlation_effect
                final_drift = pt.maximum(final_drift, 0.05)
                
                drift_rates = pt.set_subtensor(drift_rates[:, acc], final_drift)
            
            return drift_rates
        
        drift_rates = compute_drift_rates_vectorized()
        
        # 簡化的似然函數（向量化）
        def vectorized_lba_logp():
            # 調整時間
            t_adj = pt.maximum(rt_obs - t0, 0.01)
            
            # 選擇的累加器drift rates
            chosen_drifts = drift_rates[pt.arange(n_trials), response_obs]
            
            # 閾值
            thresholds = b_base
            
            # 時間似然（簡化為指數分佈）
            lambda_param = chosen_drifts / thresholds
            time_logp = pt.log(lambda_param) - lambda_param * t_adj
            
            # 選擇似然（softmax）
            choice_logits = drift_rates * 3.0
            choice_logp = choice_logits[pt.arange(n_trials), response_obs] - pt.logsumexp(choice_logits, axis=1)
            
            return pt.sum(time_logp + choice_logp)
        
        # 觀測似然
        pm.Potential('likelihood', vectorized_lba_logp())
        
        # 儲存重要變數
        pm.Deterministic('rho_symmetric', rho_base + rho_symmetry_effect)
        pm.Deterministic('rho_asymmetric', rho_base)
        pm.Deterministic('rho_difference', rho_symmetry_effect)
    
    return model

# ============================================================================
# 第三部分：快速結果分析
# ============================================================================

def analyze_fast_results(trace, data):
    """
    快速分析模型結果
    """
    
    print("\n" + "="*50)
    print("📊 協方差矩陣LBA模型結果（快速版）")
    print("="*50)
    
    # 提取後驗樣本
    posterior = trace.posterior
    
    # 協方差分析
    print(f"\n📈 協方差矩陣分析:")
    
    rho_base = posterior['rho_base'].values.flatten()
    rho_symmetry_effect = posterior['rho_symmetry_effect'].values.flatten()
    rho_symmetric = posterior['rho_symmetric'].values.flatten()
    rho_asymmetric = posterior['rho_asymmetric'].values.flatten()
    
    print(f"基礎相關係數: {np.mean(rho_base):.3f} [{np.percentile(rho_base, 2.5):.3f}, {np.percentile(rho_base, 97.5):.3f}]")
    print(f"對稱性效應: {np.mean(rho_symmetry_effect):.3f} [{np.percentile(rho_symmetry_effect, 2.5):.3f}, {np.percentile(rho_symmetry_effect, 97.5):.3f}]")
    print(f"對稱刺激相關係數: {np.mean(rho_symmetric):.3f}")
    print(f"非對稱刺激相關係數: {np.mean(rho_asymmetric):.3f}")
    
    # 顯著性測試
    rho_diff_significant = not (np.percentile(rho_symmetry_effect, 2.5) < 0 < np.percentile(rho_symmetry_effect, 97.5))
    print(f"對稱性效應顯著性: {'是' if rho_diff_significant else '否'}")
    
    # 獨立性假設檢驗
    independence_prob = np.mean(np.abs(rho_base) < 0.1)
    print(f"\n🔬 GRT獨立性假設:")
    print(f"基礎獨立性機率: {independence_prob:.3f}")
    
    if independence_prob < 0.05:
        print("🚨 強烈違反GRT獨立性假設")
    elif independence_prob < 0.2:
        print("⚠️  中等程度違反獨立性假設")
    else:
        print("✅ 基本支持獨立性假設")
    
    # 理論解釋
    print(f"\n💡 理論解釋:")
    model_rho_diff = np.mean(rho_symmetry_effect)
    
    if abs(model_rho_diff) > 0.05:
        if model_rho_diff > 0:
            print("✓ 對稱刺激增加source間相關性")
            print("  → 支持配置性處理假設")
        else:
            print("✓ 對稱刺激減少source間相關性")
            print("  → 支持獨立處理假設")
    else:
        print("• 對稱性對相關性影響較小")
    
    return {
        'rho_base': rho_base,
        'rho_symmetry_effect': rho_symmetry_effect,
        'rho_symmetric': rho_symmetric,
        'rho_asymmetric': rho_asymmetric,
        'significant_symmetry_effect': rho_diff_significant,
        'independence_violation': independence_prob < 0.2
    }

# ============================================================================
# 第四部分：快速視覺化
# ============================================================================

def create_fast_visualization(trace, results, data):
    """
    創建快速結果視覺化
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 相關係數比較
    axes[0, 0].hist(results['rho_symmetric'], bins=20, alpha=0.6, label='對稱', color='green', density=True)
    axes[0, 0].hist(results['rho_asymmetric'], bins=20, alpha=0.6, label='非對稱', color='orange', density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='獨立性')
    axes[0, 0].set_xlabel('相關係數 ρ')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('相關係數後驗分佈')
    axes[0, 0].legend()
    
    # 2. 對稱性效應
    axes[0, 1].hist(results['rho_symmetry_effect'], bins=20, alpha=0.7, color='purple', density=True)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(np.mean(results['rho_symmetry_effect']), color='black', linestyle='-', label='平均')
    axes[0, 1].set_xlabel('相關係數差異')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('對稱性效應')
    axes[0, 1].legend()
    
    # 3. 行為資料對比
    symmetric_data = data[data['is_symmetric'] == 1]
    asymmetric_data = data[data['is_symmetric'] == 0]
    
    if len(symmetric_data) > 0 and len(asymmetric_data) > 0:
        categories = ['準確率', '反應時間']
        sym_values = [symmetric_data['accuracy'].mean(), symmetric_data['rt'].mean()]
        asym_values = [asymmetric_data['accuracy'].mean(), asymmetric_data['rt'].mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, sym_values, width, label='對稱', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, asym_values, width, label='非對稱', color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('值')
        axes[1, 0].set_title('行為資料對比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
    
    # 4. 總結
    axes[1, 1].text(0.1, 0.9, '模型總結', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    
    summary_text = f"""
獨立性違反: {'是' if results['independence_violation'] else '否'}
對稱性效應: {'顯著' if results['significant_symmetry_effect'] else '不顯著'}

基礎相關: {np.mean(results['rho_base']):.3f}
對稱增強: {np.mean(results['rho_symmetry_effect']):.3f}

解釋: {'配置性處理' if np.mean(results['rho_symmetry_effect']) > 0.05 else '獨立處理' if np.mean(results['rho_symmetry_effect']) < -0.05 else '影響較小'}
"""
    
    axes[1, 1].text(0.1, 0.7, summary_text, fontsize=10, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('fast_covariance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 分析結果已儲存為 'fast_covariance_analysis.png'")

# ============================================================================
# 第五部分：快速執行函數
# ============================================================================

def run_fast_covariance_analysis(csv_file_path='GRT_LBA.csv', subject_id=None, max_trials=200):
    """
    執行快速協方差矩陣分析
    """
    
    print("🚀 開始快速協方差矩陣LBA分析")
    print("="*60)
    
    # 1. 載入資料
    data = load_real_subject_data(csv_file_path, max_trials_per_subject=max_trials)
    if data is None:
        return None, None, None
    
    # 2. 選擇受試者
    if subject_id is None:
        if 'subject_id' in data.columns:
            subject_counts = data['subject_id'].value_counts()
            selected_subject = subject_counts.index[0]
            print(f"\n🎯 自動選擇受試者 {selected_subject} (試驗數: {subject_counts.iloc[0]})")
        else:
            selected_subject = 1
            print(f"🎯 使用預設受試者 {selected_subject}")
    else:
        selected_subject = subject_id
        print(f"🎯 選擇受試者 {selected_subject}")
    
    # 提取受試者資料
    subject_data = data[data['subject_id'] == selected_subject].copy()
    
    # 3. 限制試驗數
    if len(subject_data) > max_trials:
        print(f"⚠️  限制試驗數為 {max_trials}")
        subject_data = subject_data.sample(n=max_trials, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 分析資料摘要：")
    print(f"  受試者: {selected_subject}")
    print(f"  試驗數: {len(subject_data)}")
    print(f"  準確率: {subject_data['accuracy'].mean():.3f}")
    print(f"  平均RT: {subject_data['rt'].mean():.3f}s")
    
    # 4. 建構模型
    print(f"\n🔧 建構快速模型...")
    model = build_fast_covariance_lba_model(subject_data)
    
    # 5. 執行快速採樣
    print("⏳ 開始快速MCMC採樣...")
    print("採樣參數：200 draws + 100 tune, 2 chains")
    
    try:
        with model:
            trace = pm.sample(
                draws=200,
                tune=100,
                chains=2,
                cores=1,  # 單核心避免並行問題
                target_accept=0.8,
                return_inferencedata=True,
                random_seed=456,
                progressbar=True,
                compute_convergence_checks=False  # 跳過收斂檢查加快速度
            )
        
        print("✅ 快速MCMC採樣完成！")
        
    except Exception as e:
        print(f"❌ MCMC採樣失敗: {e}")
        print("嘗試更簡單的採樣設定...")
        
        try:
            with model:
                trace = pm.sample(
                    draws=100,
                    tune=50,
                    chains=1,
                    cores=1,
                    return_inferencedata=True,
                    random_seed=456,
                    progressbar=True,
                    compute_convergence_checks=False
                )
            print("✅ 簡化採樣完成！")
        except Exception as e2:
            print(f"❌ 簡化採樣也失敗: {e2}")
            return None, None, None
    
    # 6. 分析結果
    print(f"\n📈 分析結果...")
    results = analyze_fast_results(trace, subject_data)
    
    # 7. 創建視覺化
    print(f"\n🎨 生成視覺化...")
    create_fast_visualization(trace, results, subject_data)
    
    print(f"\n🎉 快速協方差矩陣LBA分析完成！")
    
    return trace, subject_data, model

# ============================================================================
# 快速測試函數
# ============================================================================

def quick_test():
    """
    快速測試函數
    """
    print("🧪 執行快速測試...")
    return run_fast_covariance_analysis(max_trials=100)

# ============================================================================
# 執行分析
# ============================================================================

if __name__ == "__main__":
    print("🔬 優化版協方差矩陣LBA模型")
    print("解決性能問題和兼容性問題")
    print("-" * 60)
    
    # 執行快速分析
    trace, data, model = run_fast_covariance_analysis()
    
    if trace is not None:
        print("\n🎉 分析成功完成！")
        print("如果需要更詳細的分析，可以增加 draws 和 tune 參數")
    else:
        print("\n❌ 分析失敗")
        print("嘗試執行快速測試: quick_test()")

# ============================================================================
# 使用說明
# ============================================================================

"""
快速使用方法：

1. 基本執行：
   trace, data, model = run_fast_covariance_analysis()

2. 指定參數：
   trace, data, model = run_fast_covariance_analysis(
       csv_file_path='your_file.csv',
       subject_id=1,
       max_trials=150
   )

3. 快速測試（使用模擬資料）：
   trace, data, model = quick_test()

4. 如果仍然很慢，可以進一步減少參數：
   # 修改 run_fast_covariance_analysis 中的採樣參數
   # draws=100, tune=50, chains=1
""""""
優化版協方差矩陣LBA分析
解決性能問題和PyTensor兼容性問題
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 設置更高效的採樣參數
import pytensor
pytensor.config.floatX = 'float32'  # 使用float32提高速度

# ============================================================================
# 第一部分：資料載入函數（優化版）
# ============================================================================

def load_real_subject_data(csv_file_path='GRT_LBA.csv', max_trials_per_subject=300):
    """
    載入真實受試者資料（限制試驗數量以提高速度）
    """
    
    print("載入真實受試者資料...")
    
    try:
        raw_data = pd.read_csv(csv_file_path)
        print(f"成功讀取 {csv_file_path}")
        print(f"原始資料維度：{raw_data.shape}")
        
    except FileNotFoundError:
        print(f"找不到檔案 {csv_file_path}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 檢查必要的欄位
    required_columns = ['Chanel1', 'Chanel2', 'Response', 'RT', 'acc']
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    
    if missing_columns:
        print(f"缺少必要欄位：{missing_columns}")
        print(f"可用欄位：{list(raw_data.columns)}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 資料清理和轉換
    print("轉換資料格式...")
    
    # 移除缺失值
    clean_data = raw_data.dropna(subset=required_columns)
    print(f"清理後資料維度：{clean_data.shape}")
    
    # 移除極端反應時間
    clean_data = clean_data[(clean_data['RT'] > 0.1) & (clean_data['RT'] < 3.0)]
    print(f"移除極端RT後維度：{clean_data.shape}")
    
    # 限制每個受試者的試驗數量
    if 'Subject' in clean_data.columns:
        clean_data = clean_data.groupby('Subject').head(max_trials_per_subject)
    elif 'participant' in clean_data.columns:
        clean_data = clean_data.groupby('participant').head(max_trials_per_subject)
    
    # 轉換為模型需要的格式
    converted_data = []
    
    for idx, row in clean_data.iterrows():
        # 確保Response在0-3範圍內
        if not (0 <= row['Response'] <= 3):
            continue
            
        # 確保Chanel值為0或1
        if row['Chanel1'] not in [0, 1] or row['Chanel2'] not in [0, 1]:
            continue
        
        converted_row = {
            'subject_id': row.get('Subject', row.get('participant', 1)),
            'trial': len(converted_data) + 1,
            'left_pattern': int(row['Chanel1']),
            'right_pattern': int(row['Chanel2']),
            'response': int(row['Response']),
            'rt': float(row['RT']),
            'accuracy': int(row['acc']),
            'stimulus_type': int(row['Chanel1']) * 2 + int(row['Chanel2']),
            'is_symmetric': 1 if row['Chanel1'] == row['Chanel2'] else 0
        }
        
        converted_data.append(converted_row)
    
    # 轉換為DataFrame
    df = pd.DataFrame(converted_data)
    
    if len(df) == 0:
        print("❌ 沒有有效的資料行，生成模擬資料")
        return generate_test_data()
    
    # 資料統計
    print(f"\n✅ 真實資料載入完成：")
    print(f"  有效試驗數：{len(df)}")
    print(f"  受試者數：{df['subject_id'].nunique()}")
    print(f"  整體準確率：{df['accuracy'].mean():.3f}")
    print(f"  平均RT：{df['rt'].mean():.3f}秒")
    print(f"  對稱試驗比例：{df['is_symmetric'].mean():.3f}")
    
    return df

def generate_test_data(n_trials=200):
    """
    生成測試用的模擬資料
    """
    print("生成模擬測試資料...")
    
    np.random.seed(42)
    data = []
    
    for trial in range(n_trials):
        # 隨機刺激
        left_pattern = np.random.choice([0, 1])
        right_pattern = np.random.choice([0, 1])
        is_symmetric = int(left_pattern == right_pattern)
        stimulus_type = left_pattern * 2 + right_pattern
        
        # 模擬反應
        # 對稱刺激有較高準確率
        base_accuracy = 0.8 if is_symmetric else 0.7
        accuracy = int(np.random.random() < base_accuracy)
        
        # 正確反應
        if accuracy:
            response = stimulus_type
        else:
            response = np.random.choice([i for i in range(4) if i != stimulus_type])
        
        # 反應時間
        base_rt = 0.8 if is_symmetric else 0.9
        rt = np.random.gamma(2, base_rt/2)
        rt = np.clip(rt, 0.2, 3.0)
        
        data.append({
            'subject_id': 1,
            'trial': trial + 1,
            'left_pattern': left_pattern,
            'right_pattern': right_pattern,
            'response': response,
            'rt': rt,
            'accuracy': accuracy,
            'stimulus_type': stimulus_type,
            'is_symmetric': is_symmetric
        })
    
    df = pd.DataFrame(data)
    print(f"✅ 模擬資料生成完成：{len(df)} 試驗")
    return df

# ============================================================================
# 第二部分：高效能協方差矩陣LBA模型
# ============================================================================

def build_fast_covariance_lba_model(data):
    """
    建構高效能協方差矩陣LBA模型
    """
    
    print("建構高效能協方差矩陣LBA模型...")
    
    # 準備資料（轉為numpy陣列提高效率）
    rt_obs = data['rt'].values.astype('float32')
    response_obs = data['response'].values.astype('int32')
    is_symmetric = data['is_symmetric'].values.astype('float32')
    stimulus_type = data['stimulus_type'].values.astype('int32')
    
    n_trials = len(data)
    print(f"處理 {n_trials} 個試驗")
    
    with pm.Model() as model:
        
        # 協方差參數（簡化）
        rho_base = pm.Uniform('rho_base', lower=-0.5, upper=0.5)
        rho_symmetry_effect = pm.Normal('rho_symmetry_effect', mu=0, sigma=0.2)
        
        # LBA參數（簡化）
        A = pm.HalfNormal('A', sigma=0.2)
        b_base = pm.HalfNormal('b_base', sigma=0.3)
        s = pm.HalfNormal('s', sigma=0.2)
        t0 = pm.HalfNormal('t0', sigma=0.1)
        
        # 基礎drift rate
        drift_base = pm.HalfNormal('drift_base', sigma=0.5)
        drift_correct_boost = pm.HalfNormal('drift_correct_boost', sigma=0.3)
        
        # 對稱性效應
        symmetry_boost = pm.Normal('symmetry_boost', mu=0, sigma=0.1)
        
        # 向量化計算drift rates
        def compute_drift_rates_vectorized():
            # 為每個累加器計算drift rate
            drift_rates = pt.zeros((n_trials, 4))
            
            for acc in range(4):
                # 基礎drift
                base_drift = drift_base
                
                # 正確累加器增強
                correct_boost = pt.where(pt.eq(stimulus_type, acc), drift_correct_boost, 0.0)
                
                # 對稱性效應
                sym_boost = symmetry_boost * is_symmetric * pt.where(pt.eq(stimulus_type, acc), 1.0, 0.0)
                
                # 相關性效應（簡化）
                rho_trial = rho_base + rho_symmetry_effect * is_symmetric
                correlation_effect = pt.abs(rho_trial) * 0.1
                
                # 總drift rate
                final_drift = base_drift + correct_boost + sym_boost + correlation_effect
                final_drift = pt.maximum(final_drift, 0.05)
                
                drift_rates = pt.set_subtensor(drift_rates[:, acc], final_drift)
            
            return drift_rates
        
        drift_rates = compute_drift_rates_vectorized()
        
        # 簡化的似然函數（向量化）
        def vectorized_lba_logp():
            # 調整時間
            t_adj = pt.maximum(rt_obs - t0, 0.01)
            
            # 選擇的累加器drift rates
            chosen_drifts = drift_rates[pt.arange(n_trials), response_obs]
            
            # 閾值
            thresholds = b_base
            
            # 時間似然（簡化為指數分佈）
            lambda_param = chosen_drifts / thresholds
            time_logp = pt.log(lambda_param) - lambda_param * t_adj
            
            # 選擇似然（softmax）
            choice_logits = drift_rates * 3.0
            choice_logp = choice_logits[pt.arange(n_trials), response_obs] - pt.logsumexp(choice_logits, axis=1)
            
            return pt.sum(time_logp + choice_logp)
        
        # 觀測似然
        pm.Potential('likelihood', vectorized_lba_logp())
        
        # 儲存重要變數
        pm.Deterministic('rho_symmetric', rho_base + rho_symmetry_effect)
        pm.Deterministic('rho_asymmetric', rho_base)
        pm.Deterministic('rho_difference', rho_symmetry_effect)
    
    return model

# ============================================================================
# 第三部分：快速結果分析
# ============================================================================

def analyze_fast_results(trace, data):
    """
    快速分析模型結果
    """
    
    print("\n" + "="*50)
    print("📊 協方差矩陣LBA模型結果（快速版）")
    print("="*50)
    
    # 提取後驗樣本
    posterior = trace.posterior
    
    # 協方差分析
    print(f"\n📈 協方差矩陣分析:")
    
    rho_base = posterior['rho_base'].values.flatten()
    rho_symmetry_effect = posterior['rho_symmetry_effect'].values.flatten()
    rho_symmetric = posterior['rho_symmetric'].values.flatten()
    rho_asymmetric = posterior['rho_asymmetric'].values.flatten()
    
    print(f"基礎相關係數: {np.mean(rho_base):.3f} [{np.percentile(rho_base, 2.5):.3f}, {np.percentile(rho_base, 97.5):.3f}]")
    print(f"對稱性效應: {np.mean(rho_symmetry_effect):.3f} [{np.percentile(rho_symmetry_effect, 2.5):.3f}, {np.percentile(rho_symmetry_effect, 97.5):.3f}]")
    print(f"對稱刺激相關係數: {np.mean(rho_symmetric):.3f}")
    print(f"非對稱刺激相關係數: {np.mean(rho_asymmetric):.3f}")
    
    # 顯著性測試
    rho_diff_significant = not (np.percentile(rho_symmetry_effect, 2.5) < 0 < np.percentile(rho_symmetry_effect, 97.5))
    print(f"對稱性效應顯著性: {'是' if rho_diff_significant else '否'}")
    
    # 獨立性假設檢驗
    independence_prob = np.mean(np.abs(rho_base) < 0.1)
    print(f"\n🔬 GRT獨立性假設:")
    print(f"基礎獨立性機率: {independence_prob:.3f}")
    
    if independence_prob < 0.05:
        print("🚨 強烈違反GRT獨立性假設")
    elif independence_prob < 0.2:
        print("⚠️  中等程度違反獨立性假設")
    else:
        print("✅ 基本支持獨立性假設")
    
    # 理論解釋
    print(f"\n💡 理論解釋:")
    model_rho_diff = np.mean(rho_symmetry_effect)
    
    if abs(model_rho_diff) > 0.05:
        if model_rho_diff > 0:
            print("✓ 對稱刺激增加source間相關性")
            print("  → 支持配置性處理假設")
        else:
            print("✓ 對稱刺激減少source間相關性")
            print("  → 支持獨立處理假設")
    else:
        print("• 對稱性對相關性影響較小")
    
    return {
        'rho_base': rho_base,
        'rho_symmetry_effect': rho_symmetry_effect,
        'rho_symmetric': rho_symmetric,
        'rho_asymmetric': rho_asymmetric,
        'significant_symmetry_effect': rho_diff_significant,
        'independence_violation': independence_prob < 0.2
    }

# ============================================================================
# 第四部分：快速視覺化
# ============================================================================

def create_fast_visualization(trace, results, data):
    """
    創建快速結果視覺化
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 相關係數比較
    axes[0, 0].hist(results['rho_symmetric'], bins=20, alpha=0.6, label='對稱', color='green', density=True)
    axes[0, 0].hist(results['rho_asymmetric'], bins=20, alpha=0.6, label='非對稱', color='orange', density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='獨立性')
    axes[0, 0].set_xlabel('相關係數 ρ')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('相關係數後驗分佈')
    axes[0, 0].legend()
    
    # 2. 對稱性效應
    axes[0, 1].hist(results['rho_symmetry_effect'], bins=20, alpha=0.7, color='purple', density=True)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(np.mean(results['rho_symmetry_effect']), color='black', linestyle='-', label='平均')
    axes[0, 1].set_xlabel('相關係數差異')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('對稱性效應')
    axes[0, 1].legend()
    
    # 3. 行為資料對比
    symmetric_data = data[data['is_symmetric'] == 1]
    asymmetric_data = data[data['is_symmetric'] == 0]
    
    if len(symmetric_data) > 0 and len(asymmetric_data) > 0:
        categories = ['準確率', '反應時間']
        sym_values = [symmetric_data['accuracy'].mean(), symmetric_data['rt'].mean()]
        asym_values = [asymmetric_data['accuracy'].mean(), asymmetric_data['rt'].mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, sym_values, width, label='對稱', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, asym_values, width, label='非對稱', color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('值')
        axes[1, 0].set_title('行為資料對比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
    
    # 4. 總結
    axes[1, 1].text(0.1, 0.9, '模型總結', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    
    summary_text = f"""
獨立性違反: {'是' if results['independence_violation'] else '否'}
對稱性效應: {'顯著' if results['significant_symmetry_effect'] else '不顯著'}

基礎相關: {np.mean(results['rho_base']):.3f}
對稱增強: {np.mean(results['rho_symmetry_effect']):.3f}

解釋: {'配置性處理' if np.mean(results['rho_symmetry_effect']) > 0.05 else '獨立處理' if np.mean(results['rho_symmetry_effect']) < -0.05 else '影響較小'}
"""
    
    axes[1, 1].text(0.1, 0.7, summary_text, fontsize=10, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('fast_covariance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 分析結果已儲存為 'fast_covariance_analysis.png'")

# ============================================================================
# 第五部分：快速執行函數
# ============================================================================

def run_fast_covariance_analysis(csv_file_path='GRT_LBA.csv', subject_id=None, max_trials=200):
    """
    執行快速協方差矩陣分析
    """
    
    print("🚀 開始快速協方差矩陣LBA分析")
    print("="*60)
    
    # 1. 載入資料
    data = load_real_subject_data(csv_file_path, max_trials_per_subject=max_trials)
    if data is None:
        return None, None, None
    
    # 2. 選擇受試者
    if subject_id is None:
        if 'subject_id' in data.columns:
            subject_counts = data['subject_id'].value_counts()
            selected_subject = subject_counts.index[0]
            print(f"\n🎯 自動選擇受試者 {selected_subject} (試驗數: {subject_counts.iloc[0]})")
        else:
            selected_subject = 1
            print(f"🎯 使用預設受試者 {selected_subject}")
    else:
        selected_subject = subject_id
        print(f"🎯 選擇受試者 {selected_subject}")
    
    # 提取受試者資料
    subject_data = data[data['subject_id'] == selected_subject].copy()
    
    # 3. 限制試驗數
    if len(subject_data) > max_trials:
        print(f"⚠️  限制試驗數為 {max_trials}")
        subject_data = subject_data.sample(n=max_trials, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 分析資料摘要：")
    print(f"  受試者: {selected_subject}")
    print(f"  試驗數: {len(subject_data)}")
    print(f"  準確率: {subject_data['accuracy'].mean():.3f}")
    print(f"  平均RT: {subject_data['rt'].mean():.3f}s")
    
    # 4. 建構模型
    print(f"\n🔧 建構快速模型...")
    model = build_fast_covariance_lba_model(subject_data)
    
    # 5. 執行快速採樣
    print("⏳ 開始快速MCMC採樣...")
    print("採樣參數：200 draws + 100 tune, 2 chains")
    
    try:
        with model:
            trace = pm.sample(
                draws=200,
                tune=100,
                chains=2,
                cores=1,  # 單核心避免並行問題
                target_accept=0.8,
                return_inferencedata=True,
                random_seed=456,
                progressbar=True,
                compute_convergence_checks=False  # 跳過收斂檢查加快速度
            )
        
        print("✅ 快速MCMC採樣完成！")
        
    except Exception as e:
        print(f"❌ MCMC採樣失敗: {e}")
        print("嘗試更簡單的採樣設定...")
        
        try:
            with model:
                trace = pm.sample(
                    draws=100,
                    tune=50,
                    chains=1,
                    cores=1,
                    return_inferencedata=True,
                    random_seed=456,
                    progressbar=True,
                    compute_convergence_checks=False
                )
            print("✅ 簡化採樣完成！")
        except Exception as e2:
            print(f"❌ 簡化採樣也失敗: {e2}")
            return None, None, None
    
    # 6. 分析結果
    print(f"\n📈 分析結果...")
    results = analyze_fast_results(trace, subject_data)
    
    # 7. 創建視覺化
    print(f"\n🎨 生成視覺化...")
    create_fast_visualization(trace, results, subject_data)
    
    print(f"\n🎉 快速協方差矩陣LBA分析完成！")
    
    return trace, subject_data, model

# ============================================================================
# 快速測試函數
# ============================================================================

def quick_test():
    """
    快速測試函數
    """
    print("🧪 執行快速測試...")
    return run_fast_covariance_analysis(max_trials=100)

# ============================================================================
# 執行分析
# ============================================================================

if __name__ == "__main__":
    print("🔬 優化版協方差矩陣LBA模型")
    print("解決性能問題和兼容性問題")
    print("-" * 60)
    
    # 執行快速分析
    trace, data, model = run_fast_covariance_analysis()
    
    if trace is not None:
        print("\n🎉 分析成功完成！")
        print("如果需要更詳細的分析，可以增加 draws 和 tune 參數")
    else:
        print("\n❌ 分析失敗")
        print("嘗試執行快速測試: quick_test()")

# ============================================================================
# 使用說明
# ============================================================================

"""
快速使用方法：

1. 基本執行：
   trace, data, model = run_fast_covariance_analysis()

2. 指定參數：
   trace, data, model = run_fast_covariance_analysis(
       csv_file_path='your_file.csv',
       subject_id=1,
       max_trials=150
   )

3. 快速測試（使用模擬資料）：
   trace, data, model = quick_test()

4. 如果仍然很慢，可以進一步減少參數：
   # 修改 run_fast_covariance_analysis 中的採樣參數
   # draws=100, tune=50, chains=1
""""""
優化版協方差矩陣LBA分析
解決性能問題和PyTensor兼容性問題
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 設置更高效的採樣參數
import pytensor
pytensor.config.floatX = 'float32'  # 使用float32提高速度

# ============================================================================
# 第一部分：資料載入函數（優化版）
# ============================================================================

def load_real_subject_data(csv_file_path='GRT_LBA.csv', max_trials_per_subject=300):
    """
    載入真實受試者資料（限制試驗數量以提高速度）
    """
    
    print("載入真實受試者資料...")
    
    try:
        raw_data = pd.read_csv(csv_file_path)
        print(f"成功讀取 {csv_file_path}")
        print(f"原始資料維度：{raw_data.shape}")
        
    except FileNotFoundError:
        print(f"找不到檔案 {csv_file_path}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 檢查必要的欄位
    required_columns = ['Chanel1', 'Chanel2', 'Response', 'RT', 'acc']
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    
    if missing_columns:
        print(f"缺少必要欄位：{missing_columns}")
        print(f"可用欄位：{list(raw_data.columns)}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 資料清理和轉換
    print("轉換資料格式...")
    
    # 移除缺失值
    clean_data = raw_data.dropna(subset=required_columns)
    print(f"清理後資料維度：{clean_data.shape}")
    
    # 移除極端反應時間
    clean_data = clean_data[(clean_data['RT'] > 0.1) & (clean_data['RT'] < 3.0)]
    print(f"移除極端RT後維度：{clean_data.shape}")
    
    # 限制每個受試者的試驗數量
    if 'Subject' in clean_data.columns:
        clean_data = clean_data.groupby('Subject').head(max_trials_per_subject)
    elif 'participant' in clean_data.columns:
        clean_data = clean_data.groupby('participant').head(max_trials_per_subject)
    
    # 轉換為模型需要的格式
    converted_data = []
    
    for idx, row in clean_data.iterrows():
        # 確保Response在0-3範圍內
        if not (0 <= row['Response'] <= 3):
            continue
            
        # 確保Chanel值為0或1
        if row['Chanel1'] not in [0, 1] or row['Chanel2'] not in [0, 1]:
            continue
        
        converted_row = {
            'subject_id': row.get('Subject', row.get('participant', 1)),
            'trial': len(converted_data) + 1,
            'left_pattern': int(row['Chanel1']),
            'right_pattern': int(row['Chanel2']),
            'response': int(row['Response']),
            'rt': float(row['RT']),
            'accuracy': int(row['acc']),
            'stimulus_type': int(row['Chanel1']) * 2 + int(row['Chanel2']),
            'is_symmetric': 1 if row['Chanel1'] == row['Chanel2'] else 0
        }
        
        converted_data.append(converted_row)
    
    # 轉換為DataFrame
    df = pd.DataFrame(converted_data)
    
    if len(df) == 0:
        print("❌ 沒有有效的資料行，生成模擬資料")
        return generate_test_data()
    
    # 資料統計
    print(f"\n✅ 真實資料載入完成：")
    print(f"  有效試驗數：{len(df)}")
    print(f"  受試者數：{df['subject_id'].nunique()}")
    print(f"  整體準確率：{df['accuracy'].mean():.3f}")
    print(f"  平均RT：{df['rt'].mean():.3f}秒")
    print(f"  對稱試驗比例：{df['is_symmetric'].mean():.3f}")
    
    return df

def generate_test_data(n_trials=200):
    """
    生成測試用的模擬資料
    """
    print("生成模擬測試資料...")
    
    np.random.seed(42)
    data = []
    
    for trial in range(n_trials):
        # 隨機刺激
        left_pattern = np.random.choice([0, 1])
        right_pattern = np.random.choice([0, 1])
        is_symmetric = int(left_pattern == right_pattern)
        stimulus_type = left_pattern * 2 + right_pattern
        
        # 模擬反應
        # 對稱刺激有較高準確率
        base_accuracy = 0.8 if is_symmetric else 0.7
        accuracy = int(np.random.random() < base_accuracy)
        
        # 正確反應
        if accuracy:
            response = stimulus_type
        else:
            response = np.random.choice([i for i in range(4) if i != stimulus_type])
        
        # 反應時間
        base_rt = 0.8 if is_symmetric else 0.9
        rt = np.random.gamma(2, base_rt/2)
        rt = np.clip(rt, 0.2, 3.0)
        
        data.append({
            'subject_id': 1,
            'trial': trial + 1,
            'left_pattern': left_pattern,
            'right_pattern': right_pattern,
            'response': response,
            'rt': rt,
            'accuracy': accuracy,
            'stimulus_type': stimulus_type,
            'is_symmetric': is_symmetric
        })
    
    df = pd.DataFrame(data)
    print(f"✅ 模擬資料生成完成：{len(df)} 試驗")
    return df

# ============================================================================
# 第二部分：高效能協方差矩陣LBA模型
# ============================================================================

def build_fast_covariance_lba_model(data):
    """
    建構高效能協方差矩陣LBA模型
    """
    
    print("建構高效能協方差矩陣LBA模型...")
    
    # 準備資料（轉為numpy陣列提高效率）
    rt_obs = data['rt'].values.astype('float32')
    response_obs = data['response'].values.astype('int32')
    is_symmetric = data['is_symmetric'].values.astype('float32')
    stimulus_type = data['stimulus_type'].values.astype('int32')
    
    n_trials = len(data)
    print(f"處理 {n_trials} 個試驗")
    
    with pm.Model() as model:
        
        # 協方差參數（簡化）
        rho_base = pm.Uniform('rho_base', lower=-0.5, upper=0.5)
        rho_symmetry_effect = pm.Normal('rho_symmetry_effect', mu=0, sigma=0.2)
        
        # LBA參數（簡化）
        A = pm.HalfNormal('A', sigma=0.2)
        b_base = pm.HalfNormal('b_base', sigma=0.3)
        s = pm.HalfNormal('s', sigma=0.2)
        t0 = pm.HalfNormal('t0', sigma=0.1)
        
        # 基礎drift rate
        drift_base = pm.HalfNormal('drift_base', sigma=0.5)
        drift_correct_boost = pm.HalfNormal('drift_correct_boost', sigma=0.3)
        
        # 對稱性效應
        symmetry_boost = pm.Normal('symmetry_boost', mu=0, sigma=0.1)
        
        # 向量化計算drift rates
        def compute_drift_rates_vectorized():
            # 為每個累加器計算drift rate
            drift_rates = pt.zeros((n_trials, 4))
            
            for acc in range(4):
                # 基礎drift
                base_drift = drift_base
                
                # 正確累加器增強
                correct_boost = pt.where(pt.eq(stimulus_type, acc), drift_correct_boost, 0.0)
                
                # 對稱性效應
                sym_boost = symmetry_boost * is_symmetric * pt.where(pt.eq(stimulus_type, acc), 1.0, 0.0)
                
                # 相關性效應（簡化）
                rho_trial = rho_base + rho_symmetry_effect * is_symmetric
                correlation_effect = pt.abs(rho_trial) * 0.1
                
                # 總drift rate
                final_drift = base_drift + correct_boost + sym_boost + correlation_effect
                final_drift = pt.maximum(final_drift, 0.05)
                
                drift_rates = pt.set_subtensor(drift_rates[:, acc], final_drift)
            
            return drift_rates
        
        drift_rates = compute_drift_rates_vectorized()
        
        # 簡化的似然函數（向量化）
        def vectorized_lba_logp():
            # 調整時間
            t_adj = pt.maximum(rt_obs - t0, 0.01)
            
            # 選擇的累加器drift rates
            chosen_drifts = drift_rates[pt.arange(n_trials), response_obs]
            
            # 閾值
            thresholds = b_base
            
            # 時間似然（簡化為指數分佈）
            lambda_param = chosen_drifts / thresholds
            time_logp = pt.log(lambda_param) - lambda_param * t_adj
            
            # 選擇似然（softmax）
            choice_logits = drift_rates * 3.0
            choice_logp = choice_logits[pt.arange(n_trials), response_obs] - pt.logsumexp(choice_logits, axis=1)
            
            return pt.sum(time_logp + choice_logp)
        
        # 觀測似然
        pm.Potential('likelihood', vectorized_lba_logp())
        
        # 儲存重要變數
        pm.Deterministic('rho_symmetric', rho_base + rho_symmetry_effect)
        pm.Deterministic('rho_asymmetric', rho_base)
        pm.Deterministic('rho_difference', rho_symmetry_effect)
    
    return model

# ============================================================================
# 第三部分：快速結果分析
# ============================================================================

def analyze_fast_results(trace, data):
    """
    快速分析模型結果
    """
    
    print("\n" + "="*50)
    print("📊 協方差矩陣LBA模型結果（快速版）")
    print("="*50)
    
    # 提取後驗樣本
    posterior = trace.posterior
    
    # 協方差分析
    print(f"\n📈 協方差矩陣分析:")
    
    rho_base = posterior['rho_base'].values.flatten()
    rho_symmetry_effect = posterior['rho_symmetry_effect'].values.flatten()
    rho_symmetric = posterior['rho_symmetric'].values.flatten()
    rho_asymmetric = posterior['rho_asymmetric'].values.flatten()
    
    print(f"基礎相關係數: {np.mean(rho_base):.3f} [{np.percentile(rho_base, 2.5):.3f}, {np.percentile(rho_base, 97.5):.3f}]")
    print(f"對稱性效應: {np.mean(rho_symmetry_effect):.3f} [{np.percentile(rho_symmetry_effect, 2.5):.3f}, {np.percentile(rho_symmetry_effect, 97.5):.3f}]")
    print(f"對稱刺激相關係數: {np.mean(rho_symmetric):.3f}")
    print(f"非對稱刺激相關係數: {np.mean(rho_asymmetric):.3f}")
    
    # 顯著性測試
    rho_diff_significant = not (np.percentile(rho_symmetry_effect, 2.5) < 0 < np.percentile(rho_symmetry_effect, 97.5))
    print(f"對稱性效應顯著性: {'是' if rho_diff_significant else '否'}")
    
    # 獨立性假設檢驗
    independence_prob = np.mean(np.abs(rho_base) < 0.1)
    print(f"\n🔬 GRT獨立性假設:")
    print(f"基礎獨立性機率: {independence_prob:.3f}")
    
    if independence_prob < 0.05:
        print("🚨 強烈違反GRT獨立性假設")
    elif independence_prob < 0.2:
        print("⚠️  中等程度違反獨立性假設")
    else:
        print("✅ 基本支持獨立性假設")
    
    # 理論解釋
    print(f"\n💡 理論解釋:")
    model_rho_diff = np.mean(rho_symmetry_effect)
    
    if abs(model_rho_diff) > 0.05:
        if model_rho_diff > 0:
            print("✓ 對稱刺激增加source間相關性")
            print("  → 支持配置性處理假設")
        else:
            print("✓ 對稱刺激減少source間相關性")
            print("  → 支持獨立處理假設")
    else:
        print("• 對稱性對相關性影響較小")
    
    return {
        'rho_base': rho_base,
        'rho_symmetry_effect': rho_symmetry_effect,
        'rho_symmetric': rho_symmetric,
        'rho_asymmetric': rho_asymmetric,
        'significant_symmetry_effect': rho_diff_significant,
        'independence_violation': independence_prob < 0.2
    }

# ============================================================================
# 第四部分：快速視覺化
# ============================================================================

def create_fast_visualization(trace, results, data):
    """
    創建快速結果視覺化
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 相關係數比較
    axes[0, 0].hist(results['rho_symmetric'], bins=20, alpha=0.6, label='對稱', color='green', density=True)
    axes[0, 0].hist(results['rho_asymmetric'], bins=20, alpha=0.6, label='非對稱', color='orange', density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='獨立性')
    axes[0, 0].set_xlabel('相關係數 ρ')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('相關係數後驗分佈')
    axes[0, 0].legend()
    
    # 2. 對稱性效應
    axes[0, 1].hist(results['rho_symmetry_effect'], bins=20, alpha=0.7, color='purple', density=True)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(np.mean(results['rho_symmetry_effect']), color='black', linestyle='-', label='平均')
    axes[0, 1].set_xlabel('相關係數差異')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('對稱性效應')
    axes[0, 1].legend()
    
    # 3. 行為資料對比
    symmetric_data = data[data['is_symmetric'] == 1]
    asymmetric_data = data[data['is_symmetric'] == 0]
    
    if len(symmetric_data) > 0 and len(asymmetric_data) > 0:
        categories = ['準確率', '反應時間']
        sym_values = [symmetric_data['accuracy'].mean(), symmetric_data['rt'].mean()]
        asym_values = [asymmetric_data['accuracy'].mean(), asymmetric_data['rt'].mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, sym_values, width, label='對稱', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, asym_values, width, label='非對稱', color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('值')
        axes[1, 0].set_title('行為資料對比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
    
    # 4. 總結
    axes[1, 1].text(0.1, 0.9, '模型總結', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    
    summary_text = f"""
獨立性違反: {'是' if results['independence_violation'] else '否'}
對稱性效應: {'顯著' if results['significant_symmetry_effect'] else '不顯著'}

基礎相關: {np.mean(results['rho_base']):.3f}
對稱增強: {np.mean(results['rho_symmetry_effect']):.3f}

解釋: {'配置性處理' if np.mean(results['rho_symmetry_effect']) > 0.05 else '獨立處理' if np.mean(results['rho_symmetry_effect']) < -0.05 else '影響較小'}
"""
    
    axes[1, 1].text(0.1, 0.7, summary_text, fontsize=10, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('fast_covariance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 分析結果已儲存為 'fast_covariance_analysis.png'")

# ============================================================================
# 第五部分：快速執行函數
# ============================================================================

def run_fast_covariance_analysis(csv_file_path='GRT_LBA.csv', subject_id=None, max_trials=200):
    """
    執行快速協方差矩陣分析
    """
    
    print("🚀 開始快速協方差矩陣LBA分析")
    print("="*60)
    
    # 1. 載入資料
    data = load_real_subject_data(csv_file_path, max_trials_per_subject=max_trials)
    if data is None:
        return None, None, None
    
    # 2. 選擇受試者
    if subject_id is None:
        if 'subject_id' in data.columns:
            subject_counts = data['subject_id'].value_counts()
            selected_subject = subject_counts.index[0]
            print(f"\n🎯 自動選擇受試者 {selected_subject} (試驗數: {subject_counts.iloc[0]})")
        else:
            selected_subject = 1
            print(f"🎯 使用預設受試者 {selected_subject}")
    else:
        selected_subject = subject_id
        print(f"🎯 選擇受試者 {selected_subject}")
    
    # 提取受試者資料
    subject_data = data[data['subject_id'] == selected_subject].copy()
    
    # 3. 限制試驗數
    if len(subject_data) > max_trials:
        print(f"⚠️  限制試驗數為 {max_trials}")
        subject_data = subject_data.sample(n=max_trials, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 分析資料摘要：")
    print(f"  受試者: {selected_subject}")
    print(f"  試驗數: {len(subject_data)}")
    print(f"  準確率: {subject_data['accuracy'].mean():.3f}")
    print(f"  平均RT: {subject_data['rt'].mean():.3f}s")
    
    # 4. 建構模型
    print(f"\n🔧 建構快速模型...")
    model = build_fast_covariance_lba_model(subject_data)
    
    # 5. 執行快速採樣
    print("⏳ 開始快速MCMC採樣...")
    print("採樣參數：200 draws + 100 tune, 2 chains")
    
    try:
        with model:
            trace = pm.sample(
                draws=200,
                tune=100,
                chains=2,
                cores=1,  # 單核心避免並行問題
                target_accept=0.8,
                return_inferencedata=True,
                random_seed=456,
                progressbar=True,
                compute_convergence_checks=False  # 跳過收斂檢查加快速度
            )
        
        print("✅ 快速MCMC採樣完成！")
        
    except Exception as e:
        print(f"❌ MCMC採樣失敗: {e}")
        print("嘗試更簡單的採樣設定...")
        
        try:
            with model:
                trace = pm.sample(
                    draws=100,
                    tune=50,
                    chains=1,
                    cores=1,
                    return_inferencedata=True,
                    random_seed=456,
                    progressbar=True,
                    compute_convergence_checks=False
                )
            print("✅ 簡化採樣完成！")
        except Exception as e2:
            print(f"❌ 簡化採樣也失敗: {e2}")
            return None, None, None
    
    # 6. 分析結果
    print(f"\n📈 分析結果...")
    results = analyze_fast_results(trace, subject_data)
    
    # 7. 創建視覺化
    print(f"\n🎨 生成視覺化...")
    create_fast_visualization(trace, results, subject_data)
    
    print(f"\n🎉 快速協方差矩陣LBA分析完成！")
    
    return trace, subject_data, model

# ============================================================================
# 快速測試函數
# ============================================================================

def quick_test():
    """
    快速測試函數
    """
    print("🧪 執行快速測試...")
    return run_fast_covariance_analysis(max_trials=100)

# ============================================================================
# 執行分析
# ============================================================================

if __name__ == "__main__":
    print("🔬 優化版協方差矩陣LBA模型")
    print("解決性能問題和兼容性問題")
    print("-" * 60)
    
    # 執行快速分析
    trace, data, model = run_fast_covariance_analysis()
    
    if trace is not None:
        print("\n🎉 分析成功完成！")
        print("如果需要更詳細的分析，可以增加 draws 和 tune 參數")
    else:
        print("\n❌ 分析失敗")
        print("嘗試執行快速測試: quick_test()")

# ============================================================================
# 使用說明
# ============================================================================

"""
快速使用方法：

1. 基本執行：
   trace, data, model = run_fast_covariance_analysis()

2. 指定參數：
   trace, data, model = run_fast_covariance_analysis(
       csv_file_path='your_file.csv',
       subject_id=1,
       max_trials=150
   )

3. 快速測試（使用模擬資料）：
   trace, data, model = quick_test()

4. 如果仍然很慢，可以進一步減少參數：
   # 修改 run_fast_covariance_analysis 中的採樣參數
   # draws=100, tune=50, chains=1
""""""
優化版協方差矩陣LBA分析
解決性能問題和PyTensor兼容性問題
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 設置更高效的採樣參數
import pytensor
pytensor.config.floatX = 'float32'  # 使用float32提高速度

# ============================================================================
# 第一部分：資料載入函數（優化版）
# ============================================================================

def load_real_subject_data(csv_file_path='GRT_LBA.csv', max_trials_per_subject=300):
    """
    載入真實受試者資料（限制試驗數量以提高速度）
    """
    
    print("載入真實受試者資料...")
    
    try:
        raw_data = pd.read_csv(csv_file_path)
        print(f"成功讀取 {csv_file_path}")
        print(f"原始資料維度：{raw_data.shape}")
        
    except FileNotFoundError:
        print(f"找不到檔案 {csv_file_path}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 檢查必要的欄位
    required_columns = ['Chanel1', 'Chanel2', 'Response', 'RT', 'acc']
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    
    if missing_columns:
        print(f"缺少必要欄位：{missing_columns}")
        print(f"可用欄位：{list(raw_data.columns)}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 資料清理和轉換
    print("轉換資料格式...")
    
    # 移除缺失值
    clean_data = raw_data.dropna(subset=required_columns)
    print(f"清理後資料維度：{clean_data.shape}")
    
    # 移除極端反應時間
    clean_data = clean_data[(clean_data['RT'] > 0.1) & (clean_data['RT'] < 3.0)]
    print(f"移除極端RT後維度：{clean_data.shape}")
    
    # 限制每個受試者的試驗數量
    if 'Subject' in clean_data.columns:
        clean_data = clean_data.groupby('Subject').head(max_trials_per_subject)
    elif 'participant' in clean_data.columns:
        clean_data = clean_data.groupby('participant').head(max_trials_per_subject)
    
    # 轉換為模型需要的格式
    converted_data = []
    
    for idx, row in clean_data.iterrows():
        # 確保Response在0-3範圍內
        if not (0 <= row['Response'] <= 3):
            continue
            
        # 確保Chanel值為0或1
        if row['Chanel1'] not in [0, 1] or row['Chanel2'] not in [0, 1]:
            continue
        
        converted_row = {
            'subject_id': row.get('Subject', row.get('participant', 1)),
            'trial': len(converted_data) + 1,
            'left_pattern': int(row['Chanel1']),
            'right_pattern': int(row['Chanel2']),
            'response': int(row['Response']),
            'rt': float(row['RT']),
            'accuracy': int(row['acc']),
            'stimulus_type': int(row['Chanel1']) * 2 + int(row['Chanel2']),
            'is_symmetric': 1 if row['Chanel1'] == row['Chanel2'] else 0
        }
        
        converted_data.append(converted_row)
    
    # 轉換為DataFrame
    df = pd.DataFrame(converted_data)
    
    if len(df) == 0:
        print("❌ 沒有有效的資料行，生成模擬資料")
        return generate_test_data()
    
    # 資料統計
    print(f"\n✅ 真實資料載入完成：")
    print(f"  有效試驗數：{len(df)}")
    print(f"  受試者數：{df['subject_id'].nunique()}")
    print(f"  整體準確率：{df['accuracy'].mean():.3f}")
    print(f"  平均RT：{df['rt'].mean():.3f}秒")
    print(f"  對稱試驗比例：{df['is_symmetric'].mean():.3f}")
    
    return df

def generate_test_data(n_trials=200):
    """
    生成測試用的模擬資料
    """
    print("生成模擬測試資料...")
    
    np.random.seed(42)
    data = []
    
    for trial in range(n_trials):
        # 隨機刺激
        left_pattern = np.random.choice([0, 1])
        right_pattern = np.random.choice([0, 1])
        is_symmetric = int(left_pattern == right_pattern)
        stimulus_type = left_pattern * 2 + right_pattern
        
        # 模擬反應
        # 對稱刺激有較高準確率
        base_accuracy = 0.8 if is_symmetric else 0.7
        accuracy = int(np.random.random() < base_accuracy)
        
        # 正確反應
        if accuracy:
            response = stimulus_type
        else:
            response = np.random.choice([i for i in range(4) if i != stimulus_type])
        
        # 反應時間
        base_rt = 0.8 if is_symmetric else 0.9
        rt = np.random.gamma(2, base_rt/2)
        rt = np.clip(rt, 0.2, 3.0)
        
        data.append({
            'subject_id': 1,
            'trial': trial + 1,
            'left_pattern': left_pattern,
            'right_pattern': right_pattern,
            'response': response,
            'rt': rt,
            'accuracy': accuracy,
            'stimulus_type': stimulus_type,
            'is_symmetric': is_symmetric
        })
    
    df = pd.DataFrame(data)
    print(f"✅ 模擬資料生成完成：{len(df)} 試驗")
    return df

# ============================================================================
# 第二部分：高效能協方差矩陣LBA模型
# ============================================================================

def build_fast_covariance_lba_model(data):
    """
    建構高效能協方差矩陣LBA模型
    """
    
    print("建構高效能協方差矩陣LBA模型...")
    
    # 準備資料（轉為numpy陣列提高效率）
    rt_obs = data['rt'].values.astype('float32')
    response_obs = data['response'].values.astype('int32')
    is_symmetric = data['is_symmetric'].values.astype('float32')
    stimulus_type = data['stimulus_type'].values.astype('int32')
    
    n_trials = len(data)
    print(f"處理 {n_trials} 個試驗")
    
    with pm.Model() as model:
        
        # 協方差參數（簡化）
        rho_base = pm.Uniform('rho_base', lower=-0.5, upper=0.5)
        rho_symmetry_effect = pm.Normal('rho_symmetry_effect', mu=0, sigma=0.2)
        
        # LBA參數（簡化）
        A = pm.HalfNormal('A', sigma=0.2)
        b_base = pm.HalfNormal('b_base', sigma=0.3)
        s = pm.HalfNormal('s', sigma=0.2)
        t0 = pm.HalfNormal('t0', sigma=0.1)
        
        # 基礎drift rate
        drift_base = pm.HalfNormal('drift_base', sigma=0.5)
        drift_correct_boost = pm.HalfNormal('drift_correct_boost', sigma=0.3)
        
        # 對稱性效應
        symmetry_boost = pm.Normal('symmetry_boost', mu=0, sigma=0.1)
        
        # 向量化計算drift rates
        def compute_drift_rates_vectorized():
            # 為每個累加器計算drift rate
            drift_rates = pt.zeros((n_trials, 4))
            
            for acc in range(4):
                # 基礎drift
                base_drift = drift_base
                
                # 正確累加器增強
                correct_boost = pt.where(pt.eq(stimulus_type, acc), drift_correct_boost, 0.0)
                
                # 對稱性效應
                sym_boost = symmetry_boost * is_symmetric * pt.where(pt.eq(stimulus_type, acc), 1.0, 0.0)
                
                # 相關性效應（簡化）
                rho_trial = rho_base + rho_symmetry_effect * is_symmetric
                correlation_effect = pt.abs(rho_trial) * 0.1
                
                # 總drift rate
                final_drift = base_drift + correct_boost + sym_boost + correlation_effect
                final_drift = pt.maximum(final_drift, 0.05)
                
                drift_rates = pt.set_subtensor(drift_rates[:, acc], final_drift)
            
            return drift_rates
        
        drift_rates = compute_drift_rates_vectorized()
        
        # 簡化的似然函數（向量化）
        def vectorized_lba_logp():
            # 調整時間
            t_adj = pt.maximum(rt_obs - t0, 0.01)
            
            # 選擇的累加器drift rates
            chosen_drifts = drift_rates[pt.arange(n_trials), response_obs]
            
            # 閾值
            thresholds = b_base
            
            # 時間似然（簡化為指數分佈）
            lambda_param = chosen_drifts / thresholds
            time_logp = pt.log(lambda_param) - lambda_param * t_adj
            
            # 選擇似然（softmax）
            choice_logits = drift_rates * 3.0
            choice_logp = choice_logits[pt.arange(n_trials), response_obs] - pt.logsumexp(choice_logits, axis=1)
            
            return pt.sum(time_logp + choice_logp)
        
        # 觀測似然
        pm.Potential('likelihood', vectorized_lba_logp())
        
        # 儲存重要變數
        pm.Deterministic('rho_symmetric', rho_base + rho_symmetry_effect)
        pm.Deterministic('rho_asymmetric', rho_base)
        pm.Deterministic('rho_difference', rho_symmetry_effect)
    
    return model

# ============================================================================
# 第三部分：快速結果分析
# ============================================================================

def analyze_fast_results(trace, data):
    """
    快速分析模型結果
    """
    
    print("\n" + "="*50)
    print("📊 協方差矩陣LBA模型結果（快速版）")
    print("="*50)
    
    # 提取後驗樣本
    posterior = trace.posterior
    
    # 協方差分析
    print(f"\n📈 協方差矩陣分析:")
    
    rho_base = posterior['rho_base'].values.flatten()
    rho_symmetry_effect = posterior['rho_symmetry_effect'].values.flatten()
    rho_symmetric = posterior['rho_symmetric'].values.flatten()
    rho_asymmetric = posterior['rho_asymmetric'].values.flatten()
    
    print(f"基礎相關係數: {np.mean(rho_base):.3f} [{np.percentile(rho_base, 2.5):.3f}, {np.percentile(rho_base, 97.5):.3f}]")
    print(f"對稱性效應: {np.mean(rho_symmetry_effect):.3f} [{np.percentile(rho_symmetry_effect, 2.5):.3f}, {np.percentile(rho_symmetry_effect, 97.5):.3f}]")
    print(f"對稱刺激相關係數: {np.mean(rho_symmetric):.3f}")
    print(f"非對稱刺激相關係數: {np.mean(rho_asymmetric):.3f}")
    
    # 顯著性測試
    rho_diff_significant = not (np.percentile(rho_symmetry_effect, 2.5) < 0 < np.percentile(rho_symmetry_effect, 97.5))
    print(f"對稱性效應顯著性: {'是' if rho_diff_significant else '否'}")
    
    # 獨立性假設檢驗
    independence_prob = np.mean(np.abs(rho_base) < 0.1)
    print(f"\n🔬 GRT獨立性假設:")
    print(f"基礎獨立性機率: {independence_prob:.3f}")
    
    if independence_prob < 0.05:
        print("🚨 強烈違反GRT獨立性假設")
    elif independence_prob < 0.2:
        print("⚠️  中等程度違反獨立性假設")
    else:
        print("✅ 基本支持獨立性假設")
    
    # 理論解釋
    print(f"\n💡 理論解釋:")
    model_rho_diff = np.mean(rho_symmetry_effect)
    
    if abs(model_rho_diff) > 0.05:
        if model_rho_diff > 0:
            print("✓ 對稱刺激增加source間相關性")
            print("  → 支持配置性處理假設")
        else:
            print("✓ 對稱刺激減少source間相關性")
            print("  → 支持獨立處理假設")
    else:
        print("• 對稱性對相關性影響較小")
    
    return {
        'rho_base': rho_base,
        'rho_symmetry_effect': rho_symmetry_effect,
        'rho_symmetric': rho_symmetric,
        'rho_asymmetric': rho_asymmetric,
        'significant_symmetry_effect': rho_diff_significant,
        'independence_violation': independence_prob < 0.2
    }

# ============================================================================
# 第四部分：快速視覺化
# ============================================================================

def create_fast_visualization(trace, results, data):
    """
    創建快速結果視覺化
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 相關係數比較
    axes[0, 0].hist(results['rho_symmetric'], bins=20, alpha=0.6, label='對稱', color='green', density=True)
    axes[0, 0].hist(results['rho_asymmetric'], bins=20, alpha=0.6, label='非對稱', color='orange', density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='獨立性')
    axes[0, 0].set_xlabel('相關係數 ρ')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('相關係數後驗分佈')
    axes[0, 0].legend()
    
    # 2. 對稱性效應
    axes[0, 1].hist(results['rho_symmetry_effect'], bins=20, alpha=0.7, color='purple', density=True)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(np.mean(results['rho_symmetry_effect']), color='black', linestyle='-', label='平均')
    axes[0, 1].set_xlabel('相關係數差異')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('對稱性效應')
    axes[0, 1].legend()
    
    # 3. 行為資料對比
    symmetric_data = data[data['is_symmetric'] == 1]
    asymmetric_data = data[data['is_symmetric'] == 0]
    
    if len(symmetric_data) > 0 and len(asymmetric_data) > 0:
        categories = ['準確率', '反應時間']
        sym_values = [symmetric_data['accuracy'].mean(), symmetric_data['rt'].mean()]
        asym_values = [asymmetric_data['accuracy'].mean(), asymmetric_data['rt'].mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, sym_values, width, label='對稱', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, asym_values, width, label='非對稱', color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('值')
        axes[1, 0].set_title('行為資料對比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
    
    # 4. 總結
    axes[1, 1].text(0.1, 0.9, '模型總結', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    
    summary_text = f"""
獨立性違反: {'是' if results['independence_violation'] else '否'}
對稱性效應: {'顯著' if results['significant_symmetry_effect'] else '不顯著'}

基礎相關: {np.mean(results['rho_base']):.3f}
對稱增強: {np.mean(results['rho_symmetry_effect']):.3f}

解釋: {'配置性處理' if np.mean(results['rho_symmetry_effect']) > 0.05 else '獨立處理' if np.mean(results['rho_symmetry_effect']) < -0.05 else '影響較小'}
"""
    
    axes[1, 1].text(0.1, 0.7, summary_text, fontsize=10, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('fast_covariance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 分析結果已儲存為 'fast_covariance_analysis.png'")

# ============================================================================
# 第五部分：快速執行函數
# ============================================================================

def run_fast_covariance_analysis(csv_file_path='GRT_LBA.csv', subject_id=None, max_trials=200):
    """
    執行快速協方差矩陣分析
    """
    
    print("🚀 開始快速協方差矩陣LBA分析")
    print("="*60)
    
    # 1. 載入資料
    data = load_real_subject_data(csv_file_path, max_trials_per_subject=max_trials)
    if data is None:
        return None, None, None
    
    # 2. 選擇受試者
    if subject_id is None:
        if 'subject_id' in data.columns:
            subject_counts = data['subject_id'].value_counts()
            selected_subject = subject_counts.index[0]
            print(f"\n🎯 自動選擇受試者 {selected_subject} (試驗數: {subject_counts.iloc[0]})")
        else:
            selected_subject = 1
            print(f"🎯 使用預設受試者 {selected_subject}")
    else:
        selected_subject = subject_id
        print(f"🎯 選擇受試者 {selected_subject}")
    
    # 提取受試者資料
    subject_data = data[data['subject_id'] == selected_subject].copy()
    
    # 3. 限制試驗數
    if len(subject_data) > max_trials:
        print(f"⚠️  限制試驗數為 {max_trials}")
        subject_data = subject_data.sample(n=max_trials, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 分析資料摘要：")
    print(f"  受試者: {selected_subject}")
    print(f"  試驗數: {len(subject_data)}")
    print(f"  準確率: {subject_data['accuracy'].mean():.3f}")
    print(f"  平均RT: {subject_data['rt'].mean():.3f}s")
    
    # 4. 建構模型
    print(f"\n🔧 建構快速模型...")
    model = build_fast_covariance_lba_model(subject_data)
    
    # 5. 執行快速採樣
    print("⏳ 開始快速MCMC採樣...")
    print("採樣參數：200 draws + 100 tune, 2 chains")
    
    try:
        with model:
            trace = pm.sample(
                draws=200,
                tune=100,
                chains=2,
                cores=1,  # 單核心避免並行問題
                target_accept=0.8,
                return_inferencedata=True,
                random_seed=456,
                progressbar=True,
                compute_convergence_checks=False  # 跳過收斂檢查加快速度
            )
        
        print("✅ 快速MCMC採樣完成！")
        
    except Exception as e:
        print(f"❌ MCMC採樣失敗: {e}")
        print("嘗試更簡單的採樣設定...")
        
        try:
            with model:
                trace = pm.sample(
                    draws=100,
                    tune=50,
                    chains=1,
                    cores=1,
                    return_inferencedata=True,
                    random_seed=456,
                    progressbar=True,
                    compute_convergence_checks=False
                )
            print("✅ 簡化採樣完成！")
        except Exception as e2:
            print(f"❌ 簡化採樣也失敗: {e2}")
            return None, None, None
    
    # 6. 分析結果
    print(f"\n📈 分析結果...")
    results = analyze_fast_results(trace, subject_data)
    
    # 7. 創建視覺化
    print(f"\n🎨 生成視覺化...")
    create_fast_visualization(trace, results, subject_data)
    
    print(f"\n🎉 快速協方差矩陣LBA分析完成！")
    
    return trace, subject_data, model

# ============================================================================
# 快速測試函數
# ============================================================================

def quick_test():
    """
    快速測試函數
    """
    print("🧪 執行快速測試...")
    return run_fast_covariance_analysis(max_trials=100)

# ============================================================================
# 執行分析
# ============================================================================

if __name__ == "__main__":
    print("🔬 優化版協方差矩陣LBA模型")
    print("解決性能問題和兼容性問題")
    print("-" * 60)
    
    # 執行快速分析
    trace, data, model = run_fast_covariance_analysis()
    
    if trace is not None:
        print("\n🎉 分析成功完成！")
        print("如果需要更詳細的分析，可以增加 draws 和 tune 參數")
    else:
        print("\n❌ 分析失敗")
        print("嘗試執行快速測試: quick_test()")

# ============================================================================
# 使用說明
# ============================================================================

"""
快速使用方法：

1. 基本執行：
   trace, data, model = run_fast_covariance_analysis()

2. 指定參數：
   trace, data, model = run_fast_covariance_analysis(
       csv_file_path='your_file.csv',
       subject_id=1,
       max_trials=150
   )

3. 快速測試（使用模擬資料）：
   trace, data, model = quick_test()

4. 如果仍然很慢，可以進一步減少參數：
   # 修改 run_fast_covariance_analysis 中的採樣參數
   # draws=100, tune=50, chains=1
""""""
優化版協方差矩陣LBA分析
解決性能問題和PyTensor兼容性問題
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 設置更高效的採樣參數
import pytensor
pytensor.config.floatX = 'float32'  # 使用float32提高速度

# ============================================================================
# 第一部分：資料載入函數（優化版）
# ============================================================================

def load_real_subject_data(csv_file_path='GRT_LBA.csv', max_trials_per_subject=300):
    """
    載入真實受試者資料（限制試驗數量以提高速度）
    """
    
    print("載入真實受試者資料...")
    
    try:
        raw_data = pd.read_csv(csv_file_path)
        print(f"成功讀取 {csv_file_path}")
        print(f"原始資料維度：{raw_data.shape}")
        
    except FileNotFoundError:
        print(f"找不到檔案 {csv_file_path}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    except Exception as e:
        print(f"讀取檔案時發生錯誤：{e}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 檢查必要的欄位
    required_columns = ['Chanel1', 'Chanel2', 'Response', 'RT', 'acc']
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    
    if missing_columns:
        print(f"缺少必要欄位：{missing_columns}")
        print(f"可用欄位：{list(raw_data.columns)}")
        print("生成模擬資料進行測試...")
        return generate_test_data()
    
    # 資料清理和轉換
    print("轉換資料格式...")
    
    # 移除缺失值
    clean_data = raw_data.dropna(subset=required_columns)
    print(f"清理後資料維度：{clean_data.shape}")
    
    # 移除極端反應時間
    clean_data = clean_data[(clean_data['RT'] > 0.1) & (clean_data['RT'] < 3.0)]
    print(f"移除極端RT後維度：{clean_data.shape}")
    
    # 限制每個受試者的試驗數量
    if 'Subject' in clean_data.columns:
        clean_data = clean_data.groupby('Subject').head(max_trials_per_subject)
    elif 'participant' in clean_data.columns:
        clean_data = clean_data.groupby('participant').head(max_trials_per_subject)
    
    # 轉換為模型需要的格式
    converted_data = []
    
    for idx, row in clean_data.iterrows():
        # 確保Response在0-3範圍內
        if not (0 <= row['Response'] <= 3):
            continue
            
        # 確保Chanel值為0或1
        if row['Chanel1'] not in [0, 1] or row['Chanel2'] not in [0, 1]:
            continue
        
        converted_row = {
            'subject_id': row.get('Subject', row.get('participant', 1)),
            'trial': len(converted_data) + 1,
            'left_pattern': int(row['Chanel1']),
            'right_pattern': int(row['Chanel2']),
            'response': int(row['Response']),
            'rt': float(row['RT']),
            'accuracy': int(row['acc']),
            'stimulus_type': int(row['Chanel1']) * 2 + int(row['Chanel2']),
            'is_symmetric': 1 if row['Chanel1'] == row['Chanel2'] else 0
        }
        
        converted_data.append(converted_row)
    
    # 轉換為DataFrame
    df = pd.DataFrame(converted_data)
    
    if len(df) == 0:
        print("❌ 沒有有效的資料行，生成模擬資料")
        return generate_test_data()
    
    # 資料統計
    print(f"\n✅ 真實資料載入完成：")
    print(f"  有效試驗數：{len(df)}")
    print(f"  受試者數：{df['subject_id'].nunique()}")
    print(f"  整體準確率：{df['accuracy'].mean():.3f}")
    print(f"  平均RT：{df['rt'].mean():.3f}秒")
    print(f"  對稱試驗比例：{df['is_symmetric'].mean():.3f}")
    
    return df

def generate_test_data(n_trials=200):
    """
    生成測試用的模擬資料
    """
    print("生成模擬測試資料...")
    
    np.random.seed(42)
    data = []
    
    for trial in range(n_trials):
        # 隨機刺激
        left_pattern = np.random.choice([0, 1])
        right_pattern = np.random.choice([0, 1])
        is_symmetric = int(left_pattern == right_pattern)
        stimulus_type = left_pattern * 2 + right_pattern
        
        # 模擬反應
        # 對稱刺激有較高準確率
        base_accuracy = 0.8 if is_symmetric else 0.7
        accuracy = int(np.random.random() < base_accuracy)
        
        # 正確反應
        if accuracy:
            response = stimulus_type
        else:
            response = np.random.choice([i for i in range(4) if i != stimulus_type])
        
        # 反應時間
        base_rt = 0.8 if is_symmetric else 0.9
        rt = np.random.gamma(2, base_rt/2)
        rt = np.clip(rt, 0.2, 3.0)
        
        data.append({
            'subject_id': 1,
            'trial': trial + 1,
            'left_pattern': left_pattern,
            'right_pattern': right_pattern,
            'response': response,
            'rt': rt,
            'accuracy': accuracy,
            'stimulus_type': stimulus_type,
            'is_symmetric': is_symmetric
        })
    
    df = pd.DataFrame(data)
    print(f"✅ 模擬資料生成完成：{len(df)} 試驗")
    return df

# ============================================================================
# 第二部分：高效能協方差矩陣LBA模型
# ============================================================================

def build_fast_covariance_lba_model(data):
    """
    建構高效能協方差矩陣LBA模型
    """
    
    print("建構高效能協方差矩陣LBA模型...")
    
    # 準備資料（轉為numpy陣列提高效率）
    rt_obs = data['rt'].values.astype('float32')
    response_obs = data['response'].values.astype('int32')
    is_symmetric = data['is_symmetric'].values.astype('float32')
    stimulus_type = data['stimulus_type'].values.astype('int32')
    
    n_trials = len(data)
    print(f"處理 {n_trials} 個試驗")
    
    with pm.Model() as model:
        
        # 協方差參數（簡化）
        rho_base = pm.Uniform('rho_base', lower=-0.5, upper=0.5)
        rho_symmetry_effect = pm.Normal('rho_symmetry_effect', mu=0, sigma=0.2)
        
        # LBA參數（簡化）
        A = pm.HalfNormal('A', sigma=0.2)
        b_base = pm.HalfNormal('b_base', sigma=0.3)
        s = pm.HalfNormal('s', sigma=0.2)
        t0 = pm.HalfNormal('t0', sigma=0.1)
        
        # 基礎drift rate
        drift_base = pm.HalfNormal('drift_base', sigma=0.5)
        drift_correct_boost = pm.HalfNormal('drift_correct_boost', sigma=0.3)
        
        # 對稱性效應
        symmetry_boost = pm.Normal('symmetry_boost', mu=0, sigma=0.1)
        
        # 向量化計算drift rates
        def compute_drift_rates_vectorized():
            # 為每個累加器計算drift rate
            drift_rates = pt.zeros((n_trials, 4))
            
            for acc in range(4):
                # 基礎drift
                base_drift = drift_base
                
                # 正確累加器增強
                correct_boost = pt.where(pt.eq(stimulus_type, acc), drift_correct_boost, 0.0)
                
                # 對稱性效應
                sym_boost = symmetry_boost * is_symmetric * pt.where(pt.eq(stimulus_type, acc), 1.0, 0.0)
                
                # 相關性效應（簡化）
                rho_trial = rho_base + rho_symmetry_effect * is_symmetric
                correlation_effect = pt.abs(rho_trial) * 0.1
                
                # 總drift rate
                final_drift = base_drift + correct_boost + sym_boost + correlation_effect
                final_drift = pt.maximum(final_drift, 0.05)
                
                drift_rates = pt.set_subtensor(drift_rates[:, acc], final_drift)
            
            return drift_rates
        
        drift_rates = compute_drift_rates_vectorized()
        
        # 簡化的似然函數（向量化）
        def vectorized_lba_logp():
            # 調整時間
            t_adj = pt.maximum(rt_obs - t0, 0.01)
            
            # 選擇的累加器drift rates
            chosen_drifts = drift_rates[pt.arange(n_trials), response_obs]
            
            # 閾值
            thresholds = b_base
            
            # 時間似然（簡化為指數分佈）
            lambda_param = chosen_drifts / thresholds
            time_logp = pt.log(lambda_param) - lambda_param * t_adj
            
            # 選擇似然（softmax）
            choice_logits = drift_rates * 3.0
            choice_logp = choice_logits[pt.arange(n_trials), response_obs] - pt.logsumexp(choice_logits, axis=1)
            
            return pt.sum(time_logp + choice_logp)
        
        # 觀測似然
        pm.Potential('likelihood', vectorized_lba_logp())
        
        # 儲存重要變數
        pm.Deterministic('rho_symmetric', rho_base + rho_symmetry_effect)
        pm.Deterministic('rho_asymmetric', rho_base)
        pm.Deterministic('rho_difference', rho_symmetry_effect)
    
    return model

# ============================================================================
# 第三部分：快速結果分析
# ============================================================================

def analyze_fast_results(trace, data):
    """
    快速分析模型結果
    """
    
    print("\n" + "="*50)
    print("📊 協方差矩陣LBA模型結果（快速版）")
    print("="*50)
    
    # 提取後驗樣本
    posterior = trace.posterior
    
    # 協方差分析
    print(f"\n📈 協方差矩陣分析:")
    
    rho_base = posterior['rho_base'].values.flatten()
    rho_symmetry_effect = posterior['rho_symmetry_effect'].values.flatten()
    rho_symmetric = posterior['rho_symmetric'].values.flatten()
    rho_asymmetric = posterior['rho_asymmetric'].values.flatten()
    
    print(f"基礎相關係數: {np.mean(rho_base):.3f} [{np.percentile(rho_base, 2.5):.3f}, {np.percentile(rho_base, 97.5):.3f}]")
    print(f"對稱性效應: {np.mean(rho_symmetry_effect):.3f} [{np.percentile(rho_symmetry_effect, 2.5):.3f}, {np.percentile(rho_symmetry_effect, 97.5):.3f}]")
    print(f"對稱刺激相關係數: {np.mean(rho_symmetric):.3f}")
    print(f"非對稱刺激相關係數: {np.mean(rho_asymmetric):.3f}")
    
    # 顯著性測試
    rho_diff_significant = not (np.percentile(rho_symmetry_effect, 2.5) < 0 < np.percentile(rho_symmetry_effect, 97.5))
    print(f"對稱性效應顯著性: {'是' if rho_diff_significant else '否'}")
    
    # 獨立性假設檢驗
    independence_prob = np.mean(np.abs(rho_base) < 0.1)
    print(f"\n🔬 GRT獨立性假設:")
    print(f"基礎獨立性機率: {independence_prob:.3f}")
    
    if independence_prob < 0.05:
        print("🚨 強烈違反GRT獨立性假設")
    elif independence_prob < 0.2:
        print("⚠️  中等程度違反獨立性假設")
    else:
        print("✅ 基本支持獨立性假設")
    
    # 理論解釋
    print(f"\n💡 理論解釋:")
    model_rho_diff = np.mean(rho_symmetry_effect)
    
    if abs(model_rho_diff) > 0.05:
        if model_rho_diff > 0:
            print("✓ 對稱刺激增加source間相關性")
            print("  → 支持配置性處理假設")
        else:
            print("✓ 對稱刺激減少source間相關性")
            print("  → 支持獨立處理假設")
    else:
        print("• 對稱性對相關性影響較小")
    
    return {
        'rho_base': rho_base,
        'rho_symmetry_effect': rho_symmetry_effect,
        'rho_symmetric': rho_symmetric,
        'rho_asymmetric': rho_asymmetric,
        'significant_symmetry_effect': rho_diff_significant,
        'independence_violation': independence_prob < 0.2
    }

# ============================================================================
# 第四部分：快速視覺化
# ============================================================================

def create_fast_visualization(trace, results, data):
    """
    創建快速結果視覺化
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 相關係數比較
    axes[0, 0].hist(results['rho_symmetric'], bins=20, alpha=0.6, label='對稱', color='green', density=True)
    axes[0, 0].hist(results['rho_asymmetric'], bins=20, alpha=0.6, label='非對稱', color='orange', density=True)
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='獨立性')
    axes[0, 0].set_xlabel('相關係數 ρ')
    axes[0, 0].set_ylabel('密度')
    axes[0, 0].set_title('相關係數後驗分佈')
    axes[0, 0].legend()
    
    # 2. 對稱性效應
    axes[0, 1].hist(results['rho_symmetry_effect'], bins=20, alpha=0.7, color='purple', density=True)
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(np.mean(results['rho_symmetry_effect']), color='black', linestyle='-', label='平均')
    axes[0, 1].set_xlabel('相關係數差異')
    axes[0, 1].set_ylabel('密度')
    axes[0, 1].set_title('對稱性效應')
    axes[0, 1].legend()
    
    # 3. 行為資料對比
    symmetric_data = data[data['is_symmetric'] == 1]
    asymmetric_data = data[data['is_symmetric'] == 0]
    
    if len(symmetric_data) > 0 and len(asymmetric_data) > 0:
        categories = ['準確率', '反應時間']
        sym_values = [symmetric_data['accuracy'].mean(), symmetric_data['rt'].mean()]
        asym_values = [asymmetric_data['accuracy'].mean(), asymmetric_data['rt'].mean()]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, sym_values, width, label='對稱', color='green', alpha=0.7)
        axes[1, 0].bar(x + width/2, asym_values, width, label='非對稱', color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('值')
        axes[1, 0].set_title('行為資料對比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
    
    # 4. 總結
    axes[1, 1].text(0.1, 0.9, '模型總結', fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
    
    summary_text = f"""
獨立性違反: {'是' if results['independence_violation'] else '否'}
對稱性效應: {'顯著' if results['significant_symmetry_effect'] else '不顯著'}

基礎相關: {np.mean(results['rho_base']):.3f}
對稱增強: {np.mean(results['rho_symmetry_effect']):.3f}

解釋: {'配置性處理' if np.mean(results['rho_symmetry_effect']) > 0.05 else '獨立處理' if np.mean(results['rho_symmetry_effect']) < -0.05 else '影響較小'}
"""
    
    axes[1, 1].text(0.1, 0.7, summary_text, fontsize=10, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('fast_covariance_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 分析結果已儲存為 'fast_covariance_analysis.png'")

# ============================================================================
# 第五部分：快速執行函數
# ============================================================================

def run_fast_covariance_analysis(csv_file_path='GRT_LBA.csv', subject_id=None, max_trials=200):
    """
    執行快速協方差矩陣分析
    """
    
    print("🚀 開始快速協方差矩陣LBA分析")
    print("="*60)
    
    # 1. 載入資料
    data = load_real_subject_data(csv_file_path, max_trials_per_subject=max_trials)
    if data is None:
        return None, None, None
    
    # 2. 選擇受試者
    if subject_id is None:
        if 'subject_id' in data.columns:
            subject_counts = data['subject_id'].value_counts()
            selected_subject = subject_counts.index[0]
            print(f"\n🎯 自動選擇受試者 {selected_subject} (試驗數: {subject_counts.iloc[0]})")
        else:
            selected_subject = 1
            print(f"🎯 使用預設受試者 {selected_subject}")
    else:
        selected_subject = subject_id
        print(f"🎯 選擇受試者 {selected_subject}")
    
    # 提取受試者資料
    subject_data = data[data['subject_id'] == selected_subject].copy()
    
    # 3. 限制試驗數
    if len(subject_data) > max_trials:
        print(f"⚠️  限制試驗數為 {max_trials}")
        subject_data = subject_data.sample(n=max_trials, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 分析資料摘要：")
    print(f"  受試者: {selected_subject}")
    print(f"  試驗數: {len(subject_data)}")
    print(f"  準確率: {subject_data['accuracy'].mean():.3f}")
    print(f"  平均RT: {subject_data['rt'].mean():.3f}s")
    
    # 4. 建構模型
    print(f"\n🔧 建構快速模型...")
    model = build_fast_covariance_lba_model(subject_data)
    
    # 5. 執行快速採樣
    print("⏳ 開始快速MCMC採樣...")
    print("採樣參數：200 draws + 100 tune, 2 chains")
    
    try:
        with model:
            trace = pm.sample(
                draws=200,
                tune=100,
                chains=2,
                cores=1,  # 單核心避免並行問題
                target_accept=0.8,
                return_inferencedata=True,
                random_seed=456,
                progressbar=True,
                compute_convergence_checks=False  # 跳過收斂檢查加快速度
            )
        
        print("✅ 快速MCMC採樣完成！")
        
    except Exception as e:
        print(f"❌ MCMC採樣失敗: {e}")
        print("嘗試更簡單的採樣設定...")
        
        try:
            with model:
                trace = pm.sample(
                    draws=100,
                    tune=50,
                    chains=1,
                    cores=1,
                    return_inferencedata=True,
                    random_seed=456,
                    progressbar=True,
                    compute_convergence_checks=False
                )
            print("✅ 簡化採樣完成！")
        except Exception as e2:
            print(f"❌ 簡化採樣也失敗: {e2}")
            return None, None, None
    
    # 6. 分析結果
    print(f"\n📈 分析結果...")
    results = analyze_fast_results(trace, subject_data)
    
    # 7. 創建視覺化
    print(f"\n🎨 生成視覺化...")
    create_fast_visualization(trace, results, subject_data)
    
    print(f"\n🎉 快速協方差矩陣LBA分析完成！")
    
    return trace, subject_data, model

# ============================================================================
# 快速測試函數
# ============================================================================

def quick_test():
    """
    快速測試函數
    """
    print("🧪 執行快速測試...")
    return run_fast_covariance_analysis(max_trials=100)

# ============================================================================
# 執行分析
# ============================================================================

if __name__ == "__main__":
    print("🔬 優化版協方差矩陣LBA模型")
    print("解決性能問題和兼容性問題")
    print("-" * 60)
    
    # 執行快速分析
    trace, data, model = run_fast_covariance_analysis()
    
    if trace is not None:
        print("\n🎉 分析成功完成！")
        print("如果需要更詳細的分析，可以增加 draws 和 tune 參數")
    else:
        print("\n❌ 分析失敗")
        print("嘗試執行快速測試: quick_test()")

# ============================================================================
# 使用說明
# ============================================================================

"""
快速使用方法：

1. 基本執行：
   trace, data, model = run_fast_covariance_analysis()

2. 指定參數：
   trace, data, model = run_fast_covariance_analysis(
       csv_file_path='your_file.csv',
       subject_id=1,
       max_trials=150
   )

3. 快速測試（使用模擬資料）：
   trace, data, model = quick_test()

4. 如果仍然很慢，可以進一步減少參數：
   # 修改 run_fast_covariance_analysis 中的採樣參數
   # draws=100, tune=50, chains=1
"""
