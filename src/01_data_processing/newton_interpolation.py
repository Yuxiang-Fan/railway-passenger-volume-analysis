import pandas as pd
import numpy as np

def get_diff_quotient(x, y):
    # 计算差商表，返回对角线元素用于构造多项式
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
            
    return coef[0, :]

def calc_newton_poly(coef, x_data, target_x):
    # 根据差商和节点计算插值结果
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (target_x - x_data[n - k]) * p
    return p

def newton_interpolate_df(df, x_col, y_col, order=2):
    # 基于临近节点的局部牛顿插值，填补 NaN
    valid_df = df.dropna(subset=[y_col])
    known_x = valid_df[x_col].values
    known_y = valid_df[y_col].values
    
    missing_idx = df[df[y_col].isnull()].index
    
    for idx in missing_idx:
        xi = df.at[idx, x_col]
        
        # 寻找距离 xi 最近的 order + 1 个点进行局部插值
        dist = np.abs(known_x - xi)
        nearest_idx = np.argsort(dist)[:order + 1]
        
        sub_x = known_x[nearest_idx]
        sub_y = known_y[nearest_idx]
        
        # 保持 x 节点递增顺序
        sort_mask = np.argsort(sub_x)
        sub_x = sub_x[sort_mask]
        sub_y = sub_y[sort_mask]
        
        try:
            coef = get_diff_quotient(sub_x, sub_y)
            df.at[idx, y_col] = calc_newton_poly(coef, sub_x, xi)
        except ZeroDivisionError:
            print(f">> 节点 {xi} 差商计算除零，跳过该点。")
            continue

if __name__ == '__main__':
    data_path = '../../data/raw/data1.xlsx' 
    try:
        df = pd.read_excel(data_path)
    except FileNotFoundError:
        df = pd.read_excel('data1.xlsx')

    time_col = '时间' 
    target_cols = ['客运量', 'GDP', '周转量'] 

    for col in target_cols:
        if col in df.columns:
            newton_interpolate_df(df, time_col, col, order=2)

    out_path = '../../data/processed/data_interpolated.xlsx'
    df.to_excel(out_path, index=False)
    print(">> 缺失值局部插值完成。")
