import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

def check_stationarity(series, var_name, max_diff=4, sig_level=0.05):
    """
    对单个时间序列进行 ADF 检验。
    如果不平稳，则自动进行差分，最高支持 max_diff 阶差分。
    返回使其平稳的差分阶数、最终的 p-value 以及平稳后的序列。
    """
    current_series = series.dropna()
    
    for d in range(max_diff + 1):
        if d > 0:
            current_series = current_series.diff().dropna()
            
        result = adfuller(current_series, autolag='AIC')
        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        is_stationary = p_value < sig_level
        
        if is_stationary:
            print(f"[{var_name}] 在 {d} 阶差分后达到平稳 (p-value: {p_value:.4f} < {sig_level})")
            print(f"    └─ ADF Stat: {adf_stat:.4f}, 临界值 5%: {critical_values['5%']:.4f}")
            return d, p_value, current_series
        
    print(f"[{var_name}] 在 {max_diff} 阶差分后仍未在 {sig_level} 水平下平稳 (最后 p-value: {p_value:.4f})")
    print(f"    └─ ADF Stat: {adf_stat:.4f}, 临界值 5%: {critical_values['5%']:.4f}")
    return max_diff, p_value, current_series

if __name__ == '__main__':
    try:
        input_path = '../../data/processed/data_for_stat_tests.xlsx'
        df = pd.read_excel(input_path)
    except FileNotFoundError:
        df = pd.read_excel('chafen_4.xlsx')
        
    cols_to_test = df.columns[1:] 
    
    print("="*60)
    print("开始进行 ADF 平稳性检验及自动差分阶数判定")
    print("="*60)
    
    diff_orders = {}
    
    for col in cols_to_test:
        order, p_val, stationary_series = check_stationarity(df[col], col, max_diff=4)
        diff_orders[col] = order
        
    print("\n" + "="*60)
    print("差分阶数汇总:")
    for col, d in diff_orders.items():
        if d == 0:
            print(f" - {col}: 零阶单整 I(0) (原始序列平稳)")
        else:
            print(f" - {col}: {d} 阶单整 I({d})")