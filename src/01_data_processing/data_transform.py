import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def apply_log_transform(df, columns):
    """
    对指定列进行对数变换，以增强数据稳定性并减少异方差性
    使用 log(x + 1) 以处理可能存在的零值
    """
    df_log = df.copy()
    for col in columns:
        if (df_log[col] <= 0).any():
            print(f"警告: {col} 列包含非正数，将使用 log1p (log(1+x)) 处理。")
            df_log[col] = np.log1p(df_log[col])
        else:
            df_log[col] = np.log(df_log[col])
    return df_log

def normalize_data(df, columns):
    """
    去量纲化：使用 Min-Max 标准化将数据统一缩放到 [0, 1] 范围内
    """
    df_norm = df.copy()
    scaler = MinMaxScaler()
    df_norm[columns] = scaler.fit_transform(df_norm[columns])
    return df_norm, scaler

def standardize_data(df, columns):
    """
    Z-score 标准化：使数据均值为 0，方差为 1，主要用于 SVR 和岭回归模型
    """
    df_std = df.copy()
    scaler = StandardScaler()
    df_std[columns] = scaler.fit_transform(df_std[columns])
    return df_std, scaler

if __name__ == '__main__':
    try:
        input_path = '../../data/processed/data_interpolated.xlsx'
        df = pd.read_excel(input_path)
    except FileNotFoundError:
        df = pd.read_excel('data_interpolated.xlsx')

    feature_cols = [
        'GDP(现价):第三产业:年', 
        'GDP(现价):交通运输、仓储和邮政业:年',
        '城镇单位就业人员平均工资:年', 
        '铁路旅客周转量:年',
        '铁路客车拥有量:软卧车:年', 
        '铁路客车拥有量:软座车:年',
        '铁路客车拥有量:硬座车:年', 
        '高速铁路营业里程:年'
    ]
    target_col = ['客运量:铁路:年']
    all_cols = feature_cols + target_col

    print("正在进行去量纲化处理...")
    df_normalized, minmax_scaler = normalize_data(df, all_cols)

    print("正在进行对数变换...")
    df_log = apply_log_transform(df_normalized, all_cols)

    print("正在进行 Z-score 标准化...")
    df_final, z_scaler = standardize_data(df_log, all_cols)

    output_path_log = '../../data/processed/data_for_stat_tests.xlsx'
    output_path_final = '../../data/processed/data_transformed_final.xlsx'
    
    df_log.to_excel(output_path_log, index=False)
    df_final.to_excel(output_path_final, index=False)
    
    print(f"数据转换完成！")
    print(f"统计检验版本已保存至: {output_path_log}")
    print(f"模型训练版本已保存至: {output_path_final}")