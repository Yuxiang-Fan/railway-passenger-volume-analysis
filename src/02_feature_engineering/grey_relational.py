import os
import numpy as np
import pandas as pd

def minmax_scale(data):
    # MinMax 归一化映射到 [0, 1] 区间
    # 使用 np.ptp 直接计算极差，提高计算效率
    ptp = np.ptp(data, axis=0)
    # 工程防御：防止常数列导致的除零异常
    ptp = np.where(ptp == 0, 1e-8, ptp)
    return (data - np.min(data, axis=0)) / ptp

def calculate_gra(X, Y, rho=0.5):
    # 计算特征矩阵 X 与目标向量 Y 的关联系数
    X_norm = minmax_scale(X.values)
    Y_norm = minmax_scale(Y.values).reshape(-1, 1)

    # 利用 Numpy 广播机制计算绝对差值矩阵
    delta_mat = np.abs(X_norm - Y_norm)

    delta_min = np.min(delta_mat)
    delta_max = np.max(delta_mat)

    # 计算关联系数矩阵
    coef_mat = (delta_min + rho * delta_max) / (delta_mat + rho * delta_max)

    # 按列求均值得到各个特征的最终灰色关联度
    return np.mean(coef_mat, axis=0)

def select_features(df, target_col, rho=0.5, method='mean'):
    # 封装特征筛选流水线，支持动态阈值截断
    features = df.drop(columns=[target_col])
    target = df[target_col]
    
    grades = calculate_gra(features, target, rho)

    res_df = pd.DataFrame({
        'Feature': features.columns,
        'GRA_Score': grades
    }).sort_values(by='GRA_Score', ascending=False).reset_index(drop=True)

    # 动态阈值判定机制，避免硬编码带来的过拟合风险
    if method == 'mean':
        threshold = res_df['GRA_Score'].mean()
    elif isinstance(method, float):
        threshold = method
    else:
        threshold = 0.6  # 默认经验底线
        
    selected = res_df[res_df['GRA_Score'] >= threshold]['Feature'].tolist()
    return res_df, selected, threshold

if __name__ == '__main__':
    # 提供通用的数据接口规范，解耦特定数据集
    data_file = 'data/processed/dataset_ready.xlsx'
    
    if os.path.exists(data_file):
        print(f">> 加载数据集: {data_file}")
        df = pd.read_excel(data_file)
        
        # 假设最后一列为目标变量，实际工程中可通过传参动态指定
        target_name = df.columns[-1]
        
        # 使用均值法作为动态阈值，提高代码在未知数据集上的鲁棒性
        result_df, top_features, applied_thresh = select_features(df, target_name, method='mean')
        
        print("\n--- GRA Score Top Ranking ---")
        print(result_df.head(10))
        
        print(f"\n>> 触发动态阈值 (Method: {applied_thresh.__class__.__name__} -> mean) = {applied_thresh:.4f}")
        print(f">> 筛选出核心特征共 {len(top_features)} 个:")
        for feat in top_features:
            print(f" - {feat}")

        out_dir = 'outputs/tables'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'grey_relational_results.csv')
        
        result_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n>> GRA 评估报告已归档至: {out_path}")
    else:
        print(f">> [提示] 未找到 {data_file}。")
        print(">> 请确保原始数据已通过插值与平滑流水线，并放置在正确目录。")
