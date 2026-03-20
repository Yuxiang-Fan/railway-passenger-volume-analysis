import pandas as pd
import numpy as np
import os

def min_max_normalize(data):
    """
    对序列进行最小-最大规范化处理 (无量纲化)
    将其映射到 [0, 1] 区间，保证计算的有效性
    """
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def grey_relational_analysis(X, Y, rho=0.5):
    """
    计算灰色关联度
    X: 比较数列矩阵 (影响因素 DataFrame 或 2D ndarray)
    Y: 参考数列 (目标客运量 Series 或 1D ndarray)
    rho: 分辨系数，通常取 0.5
    """
    X_norm = min_max_normalize(X.values)
    Y_norm = min_max_normalize(Y.values).reshape(-1, 1)

    delta_matrix = np.abs(X_norm - Y_norm)

    delta_min = np.min(delta_matrix)
    delta_max = np.max(delta_matrix)

    coef_matrix = (delta_min + rho * delta_max) / (delta_matrix + rho * delta_max)

    relational_grades = np.mean(coef_matrix, axis=0)

    return relational_grades

if __name__ == '__main__':
    try:
        input_path = '../../data/processed/data_interpolated.xlsx'
        df = pd.read_excel(input_path)
    except FileNotFoundError:
        df = pd.read_excel('data_b.xlsx')

    features = df.iloc[:, 1:-1]
    target = df.iloc[:, -1]

    print("正在进行灰色关联度分析矩阵计算...")
    
    grades = grey_relational_analysis(features, target, rho=0.5)

    result_df = pd.DataFrame({
        '影响因素': features.columns,
        '灰色关联度': grades
    })
    result_df = result_df.sort_values(by='灰色关联度', ascending=False).reset_index(drop=True)

    print("\n===== 灰色关联度排名前 10 结果 =====")
    print(result_df.head(10))

    threshold = 0.628932
    selected_features = result_df[result_df['灰色关联度'] >= threshold]['影响因素'].tolist()
    
    print(f"\n筛选出关联度 >= {threshold} 的核心影响因素共 {len(selected_features)} 个:")
    for feat in selected_features:
        print(f"- {feat}")

    output_dir = '../../outputs/tables'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'grey_relational_results.csv')
    
    try:
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n计算完成！排序报表已成功保存至: {output_path}")
    except Exception as e:
        print(f"\n文件保存失败 (请检查路径是否存在): {e}")