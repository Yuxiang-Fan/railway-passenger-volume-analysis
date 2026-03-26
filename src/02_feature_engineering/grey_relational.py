import os
import numpy as np
import pandas as pd

def minmax_scale(data):
    # MinMax 归一化映射到 [0, 1] 区间
    # 使用 np.ptp (peak-to-peak) 直接计算极差，替代重复的 max - min
    return (data - np.min(data, axis=0)) / np.ptp(data, axis=0)

def calc_gra(X, Y, rho=0.5):
    # 计算特征矩阵 X 与目标向量 Y 的 GRA 关联系数
    X_norm = minmax_scale(X.values)
    Y_norm = minmax_scale(Y.values).reshape(-1, 1)

    # 利用 Numpy 广播机制计算绝对差值矩阵
    delta_mat = np.abs(X_norm - Y_norm)

    delta_min = np.min(delta_mat)
    delta_max = np.max(delta_mat)

    # 计算关联系数矩阵
    coef_mat = (delta_min + rho * delta_max) / (delta_mat + rho * delta_max)

    # 按列求均值得到各个特征的关联度
    return np.mean(coef_mat, axis=0)

if __name__ == '__main__':
    try:
        df = pd.read_excel('../../data/processed/data_interpolated.xlsx')
    except FileNotFoundError:
        df = pd.read_excel('data_b.xlsx')

    # 提取特征和目标变量
    features = df.iloc[:, 1:-1]
    target = df.iloc[:, -1]
    
    grades = calc_gra(features, target, rho=0.5)

    res_df = pd.DataFrame({
        'Feature': features.columns,
        'GRA_Score': grades
    }).sort_values(by='GRA_Score', ascending=False).reset_index(drop=True)

    print("--- GRA 排序结果 Top 10 ---")
    print(res_df.head(10))

    # 设定阈值筛选特征
    threshold = 0.6289
    selected = res_df[res_df['GRA_Score'] >= threshold]['Feature'].tolist()
    
    print(f"\n>> 筛选出 GRA_Score >= {threshold} 的特征共 {len(selected)} 个:")
    for feat in selected:
        print(f"- {feat}")

    out_dir = '../../outputs/tables'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'grey_relational_results.csv')
    
    res_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n>> GRA 结果已保存至: {out_path}")
