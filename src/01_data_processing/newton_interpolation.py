import pandas as pd
import numpy as np

def get_divided_diff_table(x, y):
    """
    计算牛顿插值的差商表
    n个点可以确定一个n-1阶多项式
    """
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
            
    return coef[0, :]

def newton_poly(coef, x_data, x):
    """
    计算牛顿多项式在点 x 处的值
    """
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

def interpolate_missing_values(df, x_col, y_col, order=2):
    """
    使用牛顿插值法填充缺失值
    order: 插值阶数。默认2阶（需要3个已知点），比线性插值更平滑
    """
    valid_df = df.dropna(subset=[y_col])
    known_x = valid_df[x_col].values
    known_y = valid_df[y_col].values
    
    missing_indices = df[df[y_col].isnull()].index
    
    for idx in missing_indices:
        xi = df.at[idx, x_col]
        
        distances = np.abs(known_x - xi)
        nearest_indices = np.argsort(distances)[:order + 1]
        
        subset_x = known_x[nearest_indices]
        subset_y = known_y[nearest_indices]
        
        sort_idx = np.argsort(subset_x)
        subset_x = subset_x[sort_idx]
        subset_y = subset_y[sort_idx]
        
        try:
            coef = get_divided_diff_table(subset_x, subset_y)
            interpolated_y = newton_poly(coef, subset_x, xi)
            df.at[idx, y_col] = interpolated_y
        except ZeroDivisionError:
            print(f"警告：在点 {xi} 处发生除零错误，可能是由于重复的 x 坐标导致。")
            continue

if __name__ == '__main__':
    try:
        data_path = '../../data/raw/data1.xlsx' 
        df = pd.read_excel(data_path)
    except FileNotFoundError:
        df = pd.read_excel('data1.xlsx')

    time_col = '时间' 
    target_cols = ['客运量', 'GDP', '周转量'] 

    for col in target_cols:
        if col in df.columns:
            print(f"正在对 {col} 进行牛顿插值处理...")
            interpolate_missing_values(df, time_col, col, order=2)

    output_path = '../../data/processed/data_interpolated.xlsx'
    df.to_excel(output_path, index=False)
    print(f"处理完成！结果已保存至: {output_path}")