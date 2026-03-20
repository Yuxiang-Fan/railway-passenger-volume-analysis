import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

warnings.filterwarnings('ignore')

def run_granger_causality(df, target_col, feature_cols, max_lag=3):
    """
    对目标变量和各个特征变量进行格兰杰因果检验
    捕获小样本情况下的过拟合异常
    """
    print(f"[{target_col}] 的格兰杰因果检验结果 (最大滞后阶数: {max_lag})\n" + "="*60)
    
    results_summary = {}

    for feature in feature_cols:
        print(f"\n---> 正在检验: [{feature}] 是否格兰杰引起 [{target_col}]")
        
        test_data = df[[target_col, feature]].dropna()
        
        try:
            test_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            
            for lag in range(1, max_lag + 1):
                p_value = test_result[lag][0]['ssr_ftest'][1]
                significance = "显著" if p_value < 0.05 else "不显著"
                print(f"     滞后阶数 {lag}: p-value = {p_value:.4f} ({significance})")
                
            results_summary[feature] = "成功计算"
            
        except Exception as e:
            error_msg = str(e)
            if "perfect fit" in error_msg.lower():
                print("     [异常捕获] 无法计算统计量。")
                print("     [原因分析] 数据样本量较小，导致 VAR 模型对数据拟合得过于完美，引发过拟合。")
                print("     [工程建议] 本数据集规模不足以支撑复杂的格兰杰因果分析，建议转用相关性分析或逐步回归降维。")
                results_summary[feature] = "完美拟合 (过拟合)"
            else:
                print(f"     [其他异常]: {error_msg}")
                results_summary[feature] = "计算异常"

    return results_summary

if __name__ == '__main__':
    try:
        input_path = '../../data/processed/data_for_stat_tests.xlsx'
        df = pd.read_excel(input_path)
    except FileNotFoundError:
        df = pd.read_excel('data_sd.xlsx')

    target_column = df.columns[-1]
    feature_columns = df.columns[1:-1]

    summary = run_granger_causality(df, target_col=target_column, feature_cols=feature_columns, max_lag=3)
    
    print("\n" + "="*60)
    print("格兰杰因果检验状态汇总:")
    for feat, status in summary.items():
        print(f" - {feat}: {status}")