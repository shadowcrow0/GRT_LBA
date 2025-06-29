# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 03:31:03 2025

@author: spt904
"""

"""
LBA Analysis - Main Analysis Runner (修復版)
整合所有分析模組的主要執行程式
"""

import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime
import argparse
from LBA_tool import sample_with_convergence_check


from LBA_tool import improved_model_comparison  # Use robust version
# 修復的模組導入
try:
    from lba_models import create_model_by_name, get_available_models
    from LBA_visualize import (
        create_sigma_comparison_plots, 
        create_comprehensive_summary_plots,
        create_model_comparison_matrix,
        create_participant_gallery
    )
    from LBA_IAM import run_accumulation_analysis
    from LBA_PPM import run_comprehensive_ppc
    
    print("✓ 所有分析模組導入成功")
except ImportError as e:
    print(f"❌ 模組導入失敗: {e}")
    print("請確保所有必要模組都可用")
    exit(1)

# 抑制警告
warnings.filterwarnings('ignore')

def fix_data_units_if_needed(data_file):
    """
    檢查並修復數據單位問題
    """
    print("🔧 檢查數據單位...")
    
    try:
        data = np.load(data_file, allow_pickle=True)
        observed_value = data['observed_value'].copy()
        
        # 檢查 RT 值範圍
        rt_mean = observed_value[:, 0].mean()
        print(f"當前 RT 平均值: {rt_mean:.3f}")
        
        if rt_mean < 10:  # 可能是秒
            print("檢測到 RT 是秒單位，轉換為毫秒...")
            observed_value[:, 0] *= 1000
            
            # 保存修正後的數據
            fixed_file = data_file.replace('.npz', '_fixed.npz')
            np.savez(fixed_file, 
                    observed_value=observed_value,
                    participant_idx=data['participant_idx'],
                    model_input_data=data['model_input_data'])
            
            print(f"✓ 修正後的數據已保存為: {fixed_file}")
            print(f"修正後 RT 平均值: {observed_value[:, 0].mean():.1f} ms")
            return fixed_file
        else:
            print("RT 單位正確，無需轉換")
            return data_file
            
    except Exception as e:
        print(f"❌ 數據修復失敗: {e}")
        return data_file

class LBAAnalysisRunner:
    """LBA 分析運行器主類 (修復版)"""
    
    def __init__(self, data_file='model_data.npz', output_base_dir='lba_analysis_results'):
        self.data_file = data_file
        self.output_base_dir = output_base_dir
        self.results_dir = None
        self.data = None
        self.participants = None
        
    def setup_analysis(self):
        """設置分析環境"""
        
        print("🔧 設置 LBA 分析環境...")
        
        # 修復數據單位
        self.data_file = fix_data_units_if_needed(self.data_file)
        
        # 創建輸出目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"{self.output_base_dir}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"結果目錄: {self.results_dir}")
        
        # 載入數據
        try:
            self.data = np.load(self.data_file, allow_pickle=True)
            observed_value = self.data['observed_value'].copy()
            participant_idx = self.data['participant_idx']
            model_input_data = self.data['model_input_data'].item()
            
            self.participants = np.unique(participant_idx)
            
            print(f"✓ 數據載入成功")
            print(f"✓ 找到 {len(self.participants)} 個參與者")
            print(f"✓ 總試驗數: {len(observed_value)}")
            print(f"✓ RT 範圍: {observed_value[:, 0].min():.1f} - {observed_value[:, 0].max():.1f} ms")
            print(f"✓ 平均準確率: {observed_value[:, 1].mean():.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 數據載入失敗: {e}")
            return False
    
    def run_single_participant_analysis(self, participant_id, models_to_fit=None):
        """運行單個參與者的完整分析 (修復版)"""
        
        if models_to_fit is None:
            models_to_fit = ['Coactive_Addition', 'Parallel_AND_Maximum']
        
        print(f"\n{'='*60}")
        print(f"分析參與者 {participant_id}")
        print('='*60)
        
        # 提取參與者數據
        observed_value = self.data['observed_value'].copy()
        participant_idx = self.data['participant_idx']
        model_input_data = self.data['model_input_data'].item()
        
        mask = participant_idx == participant_id
        participant_data = observed_value[mask]
        participant_input = {
            'left_match': model_input_data['left_match'][mask],
            'right_match': model_input_data['right_match'][mask]
        }
        
        n_trials = len(participant_data)
        print(f"試驗數: {n_trials}")
        print(f"RT 範圍: {participant_data[:, 0].min():.1f} - {participant_data[:, 0].max():.1f} ms")
        print(f"準確率: {participant_data[:, 1].mean():.3f}")
        
        if n_trials < 30:
            print(f"⚠️ 警告: 試驗數過少 ({n_trials})")
        
        # 擬合模型
        models = {}
        failed_models = []
        
        for model_name in models_to_fit:
            print(f"\n📊 擬合 {model_name}...")
            try:
                # 創建模型
                model = create_model_by_name(model_name, participant_data, participant_input)
                
                # 採樣
                from LBA_tool import sample_with_convergence_check
                # LBA_main.py (修正後)
                trace, diagnostics = sample_with_convergence_check(
                    model, 
                    max_attempts=2,
                    draws=500,  # 適度增加 draws 以獲得更多樣本
                    tune=1500,  # <-- 大幅增加 tune 的值
                    chains=4    # 使用 4 條鏈可以更好地診斷收斂問題
                )
                
                if trace is not None:
                    models[model_name] = trace
                    print(f"✓ {model_name} 擬合成功")
                    
                    # 保存診斷信息
                    if diagnostics:
                        print(f"  R-hat: {diagnostics.get('max_rhat', 'N/A'):.4f}")
                        print(f"  ESS: {diagnostics.get('min_ess', 'N/A'):.0f}")
                else:
                    failed_models.append(model_name)
                    print(f"❌ {model_name} 擬合失敗")
                
            except Exception as e:
                failed_models.append(model_name)
                print(f"❌ {model_name} 失敗: {e}")
        
        if not models:
            print("❌ 沒有模型成功擬合")
            return None
        
        print(f"\n成功擬合 {len(models)} 個模型，失敗 {len(failed_models)} 個")
        
        # 運行各種分析
        analysis_results = {}
        
        # 1. 累積分析
        print(f"\n🔬 運行累積分析...")
        try:
            accumulation_results = run_accumulation_analysis(models, participant_id, self.results_dir)
            analysis_results['accumulation'] = accumulation_results
            print("✓ 累積分析完成")
        except Exception as e:
            print(f"❌ 累積分析失敗: {e}")
        
        # 2. 後驗預測檢查
        print(f"\n🔍 運行後驗預測檢查...")
        try:
            ppc_results = run_comprehensive_ppc(models, participant_data, participant_id, self.results_dir)
            analysis_results['ppc'] = ppc_results
            print("✓ 後驗預測檢查完成")
        except Exception as e:
            print(f"❌ 後驗預測檢查失敗: {e}")
        
        # 3. 模型比較
        if len(models) > 1:
            print(f"\n📈 運行模型比較...")
            try:
                from LBA_tool import improved_model_comparison
                comparison_results = improved_model_comparison(models)
                analysis_results['comparison'] = comparison_results
                if comparison_results:
                    print(f"✓ 模型比較完成，獲勝者: {comparison_results['winner']}")
            except Exception as e:
                print(f"❌ 模型比較失敗: {e}")
        
        # 4. 創建參與者報告
        self.create_participant_report(participant_id, participant_data, models, analysis_results)
        
        print(f"✓ 參與者 {participant_id} 分析完成")
        return analysis_results
    
    def run_batch_analysis(self, max_participants=None, models_to_fit=None):
        """運行批次分析 (修復版)"""
        
        if models_to_fit is None:
            models_to_fit = ['Coactive_Addition', 'Parallel_AND_Maximum']
        
        print(f"\n🚀 開始批次分析...")
        print(f"擬合模型: {models_to_fit}")
        
        participants_to_analyze = self.participants
        if max_participants:
            participants_to_analyze = participants_to_analyze[:max_participants]
            print(f"限制為前 {max_participants} 個參與者")
        
        all_results = []
        failed_participants = []
        
        for i, pid in enumerate(participants_to_analyze):
            try:
                print(f"\n進度: {i+1}/{len(participants_to_analyze)}")
                result = self.run_single_participant_analysis(pid, models_to_fit)
                
                if result:
                    all_results.append({
                        'participant': pid,
                        'status': 'success',
                        'results': result
                    })
                else:
                    failed_participants.append({
                        'participant': pid,
                        'reason': 'no_models_fitted'
                    })
                    
            except Exception as e:
                print(f"❌ 參與者 {pid} 完全失敗: {e}")
                failed_participants.append({
                    'participant': pid,
                    'reason': str(e)
                })
        
        # 創建批次摘要
        self.create_batch_summary(all_results, failed_participants)
        
        return all_results, failed_participants
    
    def create_participant_report(self, participant_id, participant_data, models, analysis_results):
        """創建參與者詳細報告"""
        
        report_file = os.path.join(self.results_dir, f'participant_{participant_id}_detailed_report.txt')
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"詳細分析報告\n")
                f.write(f"參與者: {participant_id}\n")
                f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                # 基本統計
                f.write("基本統計\n")
                f.write("-" * 20 + "\n")
                f.write(f"總試驗數: {len(participant_data)}\n")
                f.write(f"準確率: {participant_data[:, 1].mean():.3f}\n")
                f.write(f"平均 RT: {participant_data[:, 0].mean():.1f} ms\n")
                f.write(f"RT 標準差: {participant_data[:, 0].std():.1f} ms\n")
                f.write(f"RT 範圍: {participant_data[:, 0].min():.1f} - {participant_data[:, 0].max():.1f} ms\n\n")
                
                # 模型結果
                f.write("模型擬合結果\n")
                f.write("-" * 25 + "\n")
                for model_name in models.keys():
                    f.write(f"{model_name}: 成功擬合\n")
                f.write(f"\n總計擬合模型數: {len(models)}\n\n")
                
                # 分析結果摘要
                f.write("分析結果摘要\n")
                f.write("-" * 28 + "\n")
                
                if 'accumulation' in analysis_results:
                    f.write("✓ 累積分析完成\n")
                
                if 'ppc' in analysis_results:
                    f.write("✓ 後驗預測檢查完成\n")
                
                if 'comparison' in analysis_results and analysis_results['comparison']:
                    comp_result = analysis_results['comparison']
                    f.write("✓ 模型比較完成\n")
                    f.write(f"  獲勝模型: {comp_result['winner']}\n")
                    f.write(f"  效應量: {comp_result.get('effect_size', 'N/A'):.3f}\n")
                
                f.write("\n生成文件\n")
                f.write("-" * 15 + "\n")
                f.write("此分析生成了以下文件:\n")
                f.write(f"- participant_{participant_id}_*_accumulation.png (累積圖)\n")
                f.write(f"- participant_{participant_id}_posterior_predictive_check.png\n")
                f.write(f"- participant_{participant_id}_residual_analysis.png\n")
                f.write(f"- participant_{participant_id}_qq_plots.png\n")
            
            print(f"    ✓ 詳細報告已保存: {report_file}")
            
        except Exception as e:
            print(f"    ❌ 創建報告失敗: {e}")
    
    def create_batch_summary(self, all_results, failed_participants):
        """創建批次分析摘要"""
        
        print(f"\n📋 創建批次分析摘要...")
        
        total_participants = len(all_results) + len(failed_participants)
        success_rate = len(all_results) / total_participants * 100 if total_participants > 0 else 0
        
        # 創建摘要報告
        summary_file = os.path.join(self.results_dir, 'batch_analysis_summary.txt')
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("批次分析摘要\n")
                f.write("=" * 30 + "\n")
                f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("整體統計\n")
                f.write("-" * 20 + "\n")
                f.write(f"處理的參與者總數: {total_participants}\n")
                f.write(f"成功分析: {len(all_results)}\n")
                f.write(f"失敗分析: {len(failed_participants)}\n")
                f.write(f"成功率: {success_rate:.1f}%\n\n")
                
                if failed_participants:
                    f.write("失敗的參與者\n")
                    f.write("-" * 20 + "\n")
                    for fp in failed_participants:
                        f.write(f"參與者 {fp['participant']}: {fp['reason']}\n")
                    f.write("\n")
                
                f.write("成功的參與者\n")
                f.write("-" * 25 + "\n")
                for result in all_results:
                    pid = result['participant']
                    f.write(f"參與者 {pid}: 分析完成\n")
                
                f.write(f"\n分析輸出\n")
                f.write("-" * 16 + "\n")
                f.write("每個參與者都有以下輸出文件:\n")
                f.write("- 累積圖 (信息累積軌跡)\n")
                f.write("- 後驗預測檢查圖\n")
                f.write("- 殘差分析圖\n")
                f.write("- Q-Q 圖用於分布驗證\n")
                f.write("- 摘要表格和文本報告\n")
            
            print(f"✓ 批次摘要已保存: {summary_file}")
            
        except Exception as e:
            print(f"❌ 創建批次摘要失敗: {e}")
        
        # 創建成功參與者列表
        if all_results:
            try:
                success_df = pd.DataFrame([
                    {
                        'participant': r['participant'],
                        'status': r['status']
                    } for r in all_results
                ])
                
                csv_file = os.path.join(self.results_dir, 'successful_participants.csv')
                success_df.to_csv(csv_file, index=False)
                print(f"✓ 成功參與者列表已保存: {csv_file}")
            except Exception as e:
                print(f"⚠️ 保存成功參與者列表失敗: {e}")
        
        # 創建失敗參與者列表
        if failed_participants:
            try:
                failed_df = pd.DataFrame(failed_participants)
                csv_file = os.path.join(self.results_dir, 'failed_participants.csv')
                failed_df.to_csv(csv_file, index=False)
                print(f"✓ 失敗參與者列表已保存: {csv_file}")
            except Exception as e:
                print(f"⚠️ 保存失敗參與者列表失敗: {e}")

def create_analysis_index_html(results_dir):
    """創建分析結果的HTML索引頁面"""
    
    print("創建分析索引頁面...")
    
    # 掃描結果目錄
    participant_files = {}
    general_files = []
    
    try:
        for file in os.listdir(results_dir):
            if file.startswith('participant_'):
                # 提取參與者ID
                try:
                    parts = file.split('_')
                    if len(parts) >= 2:
                        pid = parts[1]
                        if pid not in participant_files:
                            participant_files[pid] = []
                        participant_files[pid].append(file)
                except:
                    continue
            else:
                general_files.append(file)
    except Exception as e:
        print(f"⚠️ 掃描文件失敗: {e}")
        return None
    
    # 生成HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LBA 分析結果</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            .summary {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 4px solid #3498db;
            }}
            .file-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .file-card {{
                border: 1px solid #dee2e6;
                padding: 15px;
                border-radius: 8px;
                background-color: #ffffff;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .file-card h3 {{
                margin-top: 0;
                color: #2c3e50;
                font-size: 1.1em;
            }}
            .file-list {{
                list-style: none;
                padding: 0;
            }}
            .file-list li {{
                margin: 8px 0;
                padding: 5px 0;
                border-bottom: 1px solid #eee;
            }}
            .file-list a {{
                color: #3498db;
                text-decoration: none;
                font-size: 0.9em;
            }}
            .file-list a:hover {{
                color: #2980b9;
                text-decoration: underline;
            }}
            .general-files {{
                background-color: #e8f5e8;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .timestamp {{
                color: #7f8c8d;
                font-size: 0.9em;
                font-style: italic;
            }}
            .stats {{
                display: flex;
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-box {{
                background-color: #3498db;
                color: white;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
                flex: 1;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 LBA 模型分析結果</h1>
            <p class="timestamp">生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>📊 分析概覽</h2>
                <p>這個綜合分析包含線性彈道累積器 (LBA) 模型擬合:</p>
                <ul>
                    <li><strong>信息累積圖</strong> - 證據累積過程的可視化</li>
                    <li><strong>後驗預測檢查</strong> - 透過數據預測進行模型驗證</li>
                    <li><strong>殘差分析</strong> - 模型擬合質量評估</li>
                    <li><strong>Q-Q 圖</strong> - 分布對齊驗證</li>
                    <li><strong>綜合報告</strong> - 詳細統計摘要</li>
                </ul>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number">{len(participant_files)}</div>
                    <div class="stat-label">分析的參與者</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{len(general_files)}</div>
                    <div class="stat-label">摘要文件</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{sum(len(files) for files in participant_files.values())}</div>
                    <div class="stat-label">總輸出文件</div>
                </div>
            </div>
            
            <div class="general-files">
                <h2>📋 一般分析文件</h2>
                <ul class="file-list">
    """
    
    # 添加一般文件
    for file in sorted(general_files):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            html_content += f'                    <li>🖼️ <a href="{file}">{file}</a> (圖像)</li>\n'
        elif file.endswith('.csv'):
            html_content += f'                    <li>📊 <a href="{file}">{file}</a> (數據)</li>\n'
        elif file.endswith('.txt'):
            html_content += f'                    <li>📄 <a href="{file}">{file}</a> (報告)</li>\n'
        elif file.endswith('.html'):
            html_content += f'                    <li>🌐 <a href="{file}">{file}</a> (網頁)</li>\n'
        else:
            html_content += f'                    <li>📁 <a href="{file}">{file}</a></li>\n'
    
    html_content += """
                </ul>
            </div>
            
            <h2>👥 個別參與者結果</h2>
            <div class="file-grid">
    """
    
    # 添加參與者文件卡片
    for pid in sorted(participant_files.keys()):
        files = participant_files[pid]
        html_content += f"""
                <div class="file-card">
                    <h3>👤 參與者 {pid}</h3>
                    <ul class="file-list">
        """
        
        # 按文件類型排序
        image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        data_files = [f for f in files if f.endswith(('.csv', '.npz'))]
        text_files = [f for f in files if f.endswith('.txt')]
        
        # 添加圖片文件
        for file in sorted(image_files):
            if 'accumulation' in file:
                html_content += f'                        <li>🔬 <a href="{file}">累積圖</a></li>\n'
            elif 'predictive' in file:
                html_content += f'                        <li>🔍 <a href="{file}">預測檢查</a></li>\n'
            elif 'residual' in file:
                html_content += f'                        <li>📈 <a href="{file}">殘差分析</a></li>\n'
            elif 'qq' in file:
                html_content += f'                        <li>📊 <a href="{file}">Q-Q 圖</a></li>\n'
            else:
                html_content += f'                        <li>🖼️ <a href="{file}">{file}</a></li>\n'
        
        # 添加數據文件
        for file in sorted(data_files):
            html_content += f'                        <li>📊 <a href="{file}">數據文件</a></li>\n'
        
        # 添加文本文件
        for file in sorted(text_files):
            if 'report' in file:
                html_content += f'                        <li>📄 <a href="{file}">詳細報告</a></li>\n'
            elif 'summary' in file:
                html_content += f'                        <li>📋 <a href="{file}">摘要</a></li>\n'
            else:
                html_content += f'                        <li>📄 <a href="{file}">{file}</a></li>\n'
        
        html_content += """
                    </ul>
                </div>
        """
    
    html_content += """
            </div>
            
            <div class="summary">
                <h2>📖 結果導覽方式</h2>
                <ol>
                    <li><strong>從一般文件開始</strong> - 查看批次摘要和整體統計</li>
                    <li><strong>檢查個別參與者</strong> - 點擊參與者卡片查看詳細結果</li>
                    <li><strong>查看累積圖</strong> - 了解證據如何隨時間累積</li>
                    <li><strong>檢查預測檢查</strong> - 評估模型預測實際數據的程度</li>
                    <li><strong>閱讀詳細報告</strong> - 獲得綜合統計摘要</li>
                </ol>
                
                <p><strong>需要幫助？</strong> 每種分析類型都包含解釋文本和統計解釋。</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML文件
    try:
        html_file = os.path.join(results_dir, 'index.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ 分析索引頁面已創建: {html_file}")
        return html_file
    except Exception as e:
        print(f"❌ 創建索引頁面失敗: {e}")
        return None

def main():
    """主程序"""
    
    parser = argparse.ArgumentParser(description='LBA 模型分析運行器')
    parser.add_argument('--mode', choices=['single', 'batch', 'test'], default='test',
                       help='分析模式: single (單個參與者), batch (批次), 或 test (測試)')
    parser.add_argument('--participant', type=str, help='單個模式的參與者ID')
    parser.add_argument('--max-participants', type=int, help='批次模式的最大參與者數')
    parser.add_argument('--models', nargs='+', default=['Coactive_Addition', 'Parallel_AND_Maximum'],
                       help='要擬合的模型')
    parser.add_argument('--data-file', default='model_data.npz', help='輸入數據文件')
    parser.add_argument('--output-dir', default='lba_analysis_results', help='輸出目錄基礎名稱')
    
    args = parser.parse_args()
    
    print("🧠 LBA 模型分析運行器 (修復版)")
    print("=" * 50)
    print(f"模式: {args.mode}")
    print(f"模型: {args.models}")
    print(f"數據文件: {args.data_file}")
    
    # 檢查數據文件
    if not os.path.exists(args.data_file):
        print(f"❌ 數據文件未找到: {args.data_file}")
        print("請確保數據文件存在於當前目錄中。")
        return
    
    # 創建分析器
    runner = LBAAnalysisRunner(data_file=args.data_file, output_base_dir=args.output_dir)
    
    # 設置環境
    if not runner.setup_analysis():
        print("❌ 分析環境設置失敗")
        return
    
    # 根據模式運行分析
    if args.mode == 'test':
        print("\n🧪 運行測試模式...")
        print("這將分析第一個參與者作為測試。")
        
        if len(runner.participants) > 0:
            test_participant = runner.participants[0]
            print(f"測試參與者: {test_participant}")
            
            result = runner.run_single_participant_analysis(test_participant, args.models)
            
            if result:
                print("\n✅ 測試成功完成！")
                print(f"檢查結果於: {runner.results_dir}")
            else:
                print("\n❌ 測試失敗")
        else:
            print("❌ 未找到參與者")
    
    elif args.mode == 'single':
        if not args.participant:
            print("❌ 單個模式需要參與者ID")
            print("可用參與者:", runner.participants[:10], "...")
            return
        
        if args.participant in runner.participants.astype(str):
            result = runner.run_single_participant_analysis(args.participant, args.models)
            
            if result:
                print(f"\n✅ 參與者 {args.participant} 分析完成！")
            else:
                print(f"\n❌ 參與者 {args.participant} 分析失敗")
        else:
            print(f"❌ 未找到參與者 {args.participant}")
    
    elif args.mode == 'batch':
        print(f"\n🚀 運行批次分析...")
        
        if args.max_participants:
            print(f"限制為 {args.max_participants} 個參與者")
        
        all_results, failed_participants = runner.run_batch_analysis(
            max_participants=args.max_participants,
            models_to_fit=args.models
        )
        
        print(f"\n📊 批次分析摘要:")
        print(f"✅ 成功: {len(all_results)}")
        print(f"❌ 失敗: {len(failed_participants)}")
        
        if all_results:
            success_rate = len(all_results) / (len(all_results) + len(failed_participants)) * 100
            print(f"📈 成功率: {success_rate:.1f}%")
    
    # 創建索引頁面
    if runner.results_dir:
        create_analysis_index_html(runner.results_dir)
        
        print(f"\n🎉 分析完成！")
        print(f"📁 結果保存於: {runner.results_dir}")
        print(f"🌐 打開 index.html 瀏覽結果")

if __name__ == '__main__':
    main()