# -*- coding: utf-8 -*-
"""
trace_diagnostics.py - MCMC诊断：Traceplot和Rhat分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import warnings
warnings.filterwarnings('ignore')

class MCMCDiagnostics:
    """MCMC诊断工具"""
    
    def __init__(self):
        self.trace = None
        self.participant_id = None
        
    def load_trace_results(self, participant_id: int, 
                          params_file: str = "high_accuracy_parallel_and_parameters.csv") -> bool:
        """载入trace结果"""
        
        print(f"📊 载入参与者 {participant_id} 的trace结果...")
        
        try:
            # 检查是否有保存的trace文件
            import pickle
            trace_file = f"participant_{participant_id}_trace.pkl"
            
            try:
                with open(trace_file, 'rb') as f:
                    self.trace = pickle.load(f)
                self.participant_id = participant_id
                print(f"   ✅ 成功载入trace文件: {trace_file}")
                return True
            except FileNotFoundError:
                print(f"   ❌ 找不到trace文件: {trace_file}")
                print("   💡 请先运行fit_all_participants_parallel_and.py并保存trace")
                return False
                
        except Exception as e:
            print(f"❌ 载入失败: {e}")
            return False
    
    def compute_rhat_values(self) -> pd.DataFrame:
        """计算所有参数的Rhat值"""
        
        print("🔍 计算Rhat值...")
        
        if self.trace is None:
            print("❌ 请先载入trace数据")
            return pd.DataFrame()
        
        # 计算Rhat
        rhat_values = az.rhat(self.trace)
        
        # 转换为DataFrame
        rhat_data = []
        for var_name in rhat_values.data_vars:
            rhat_val = float(rhat_values[var_name].values)
            rhat_data.append({
                'parameter': var_name,
                'rhat': rhat_val,
                'converged': rhat_val < 1.1,  # 通常认为Rhat < 1.1表示收敛
                'status': 'Good' if rhat_val < 1.05 else 'Acceptable' if rhat_val < 1.1 else 'Poor'
            })
        
        rhat_df = pd.DataFrame(rhat_data)
        rhat_df = rhat_df.sort_values('rhat', ascending=False)
        
        # 打印结果
        print(f"\n📈 Rhat诊断结果:")
        print(f"   总参数数: {len(rhat_df)}")
        print(f"   收敛参数 (Rhat < 1.1): {rhat_df['converged'].sum()}")
        print(f"   未收敛参数: {(~rhat_df['converged']).sum()}")
        
        print(f"\n🔍 详细Rhat值:")
        for _, row in rhat_df.iterrows():
            status_emoji = "✅" if row['converged'] else "❌"
            print(f"   {status_emoji} {row['parameter']}: {row['rhat']:.4f} ({row['status']})")
        
        return rhat_df
    
    def create_trace_plots(self, save_path: str = None) -> plt.Figure:
        """创建traceplot"""
        
        print("🎨 创建Trace plots...")
        
        if self.trace is None:
            print("❌ 请先载入trace数据")
            return None
        
        # 获取所有参数
        var_names = list(self.trace.posterior.data_vars.keys())
        n_vars = len(var_names)
        
        # 创建subplot布局
        n_cols = 3
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 绘制每个参数的trace
        for i, var_name in enumerate(var_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # 获取trace数据
            trace_data = self.trace.posterior[var_name]
            
            # 绘制每个chain
            for chain in range(trace_data.sizes['chain']):
                chain_data = trace_data.isel(chain=chain)
                ax.plot(chain_data, alpha=0.7, label=f'Chain {chain}')
            
            ax.set_title(f'{var_name}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的subplot
        for i in range(n_vars, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"trace_plots_participant_{self.participant_id}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Trace plots已保存: {save_path}")
        
        return fig
    
    def create_posterior_plots(self, save_path: str = None) -> plt.Figure:
        """创建posterior分布图"""
        
        print("🎨 创建Posterior分布图...")
        
        if self.trace is None:
            print("❌ 请先载入trace数据")
            return None
        
        # 使用ArviZ创建posterior plot
        fig = az.plot_posterior(self.trace, figsize=(15, 12))
        
        if save_path is None:
            save_path = f"posterior_plots_participant_{self.participant_id}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Posterior plots已保存: {save_path}")
        
        return fig
    
    def create_summary_table(self) -> pd.DataFrame:
        """创建参数汇总表"""
        
        print("📋 创建参数汇总表...")
        
        if self.trace is None:
            print("❌ 请先载入trace数据")
            return pd.DataFrame()
        
        # 使用ArviZ计算汇总统计
        summary = az.summary(self.trace)
        
        print(f"\n📊 参数汇总统计:")
        print(summary.to_string())
        
        # 保存到CSV
        summary_file = f"mcmc_summary_participant_{self.participant_id}.csv"
        summary.to_csv(summary_file)
        print(f"\n💾 汇总表已保存: {summary_file}")
        
        return summary
    
    def run_complete_diagnostics(self, participant_id: int) -> dict:
        """运行完整的MCMC诊断"""
        
        print(f"🔬 执行参与者 {participant_id} 的完整MCMC诊断")
        print("="*60)
        
        # 载入数据
        if not self.load_trace_results(participant_id):
            return {}
        
        # 计算Rhat
        rhat_df = self.compute_rhat_values()
        
        # 创建trace plots
        trace_fig = self.create_trace_plots()
        
        # 创建posterior plots
        posterior_fig = self.create_posterior_plots()
        
        # 创建汇总表
        summary_df = self.create_summary_table()
        
        results = {
            'participant_id': participant_id,
            'rhat_results': rhat_df,
            'summary_stats': summary_df,
            'trace_figure': trace_fig,
            'posterior_figure': posterior_fig
        }
        
        print(f"\n✅ MCMC诊断完成!")
        return results

def create_trace_from_existing_fit(participant_id: int = 40):
    """从现有的fitting结果重新创建trace用于诊断"""
    
    print(f"🔄 为参与者 {participant_id} 重新运行fitting以获取trace...")
    
    # 重新import fitting类
    from fit_all_participants_parallel_and import ParallelANDModelFitter
    
    # 载入数据
    fitter = ParallelANDModelFitter()
    all_data = fitter.load_all_participants("GRT_LBA.csv", accuracy_threshold=0.65)
    
    if participant_id not in all_data:
        print(f"❌ 参与者 {participant_id} 不在高正确率数据中")
        return None
    
    # 重新fit来获取trace
    result = fitter.fit_participant(participant_id, all_data[participant_id], 
                                   n_samples=1000, n_tune=1000)
    
    if result and 'trace' in result:
        # 保存trace
        import pickle
        trace_file = f"participant_{participant_id}_trace.pkl"
        with open(trace_file, 'wb') as f:
            pickle.dump(result['trace'], f)
        print(f"✅ Trace已保存: {trace_file}")
        return result['trace']
    else:
        print("❌ Fitting失败")
        return None

def main():
    """主要执行函数"""
    
    print("🔍 MCMC诊断工具")
    print("="*40)
    
    # 选择参与者
    participant_id = 40  # 可以修改
    
    # 初始化诊断工具
    diagnostics = MCMCDiagnostics()
    
    # 尝试载入trace
    if not diagnostics.load_trace_results(participant_id):
        print("📦 重新运行fitting来获取trace...")
        trace = create_trace_from_existing_fit(participant_id)
        if trace is None:
            print("❌ 无法获取trace数据")
            return
    
    # 运行完整诊断
    results = diagnostics.run_complete_diagnostics(participant_id)
    
    if results:
        print(f"\n🎉 参与者 {participant_id} 的MCMC诊断完成!")
        
        # 输出关键信息
        rhat_df = results['rhat_results']
        if len(rhat_df) > 0:
            print(f"\n🎯 收敛性总结:")
            print(f"   收敛参数比例: {rhat_df['converged'].mean():.1%}")
            print(f"   最差Rhat值: {rhat_df['rhat'].max():.4f}")
            print(f"   平均Rhat值: {rhat_df['rhat'].mean():.4f}")
    else:
        print("❌ 诊断失败")

if __name__ == "__main__":
    main()