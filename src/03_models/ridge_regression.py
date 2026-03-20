import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """
    加载数据并分离特征与目标变量。
    跳过第一列(时间)，取最后一列为目标变量。
    """
    df = pd.read_excel(filepath)
    
    X = df.iloc[:, 1:-1]
    Y = df.iloc[:, -1]
    
    return X, Y, df.columns[1:-1].tolist(), df.columns[-1]

def run_ridge_pipeline(data_path, alpha=1.0, use_full_data=True):
    """
    执行岭回归的完整流水线。
    use_full_data: 默认为 True，使用全量数据拟合
    """
    X, Y, feature_names, target_name = load_and_prepare_data(data_path)
    
    if use_full_data:
        X_train, X_test = X, X
        Y_train, Y_test = Y, Y
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, Y_train)

    Y_predict = ridge_model.predict(X_test_scaled)
    mse = mean_squared_error(Y_test, Y_predict)

    print("\n====== 岭回归 (Ridge Regression) 拟合结果 ======")
    print(f"使用的正则化参数 Alpha: {alpha}")
    print("\n模型特征系数 (Coefficients):")
    for feat, coef in zip(feature_names, ridge_model.coef_):
        print(f"  - {feat}: {coef:.6f}")
    
    print(f"\n模型偏置项 (Intercept): {ridge_model.intercept_:.6f}")
    print(f"均方误差 (MSE): {mse:.10f}")

    plot_and_save_ridge_results(Y_test, Y_predict)

def plot_and_save_ridge_results(Y_true, Y_pred):
    """
    绘制真实值与预测值的拟合图表并保存
    """
    plt.figure(figsize=(10, 6))
    
    plt.scatter(Y_true, Y_pred, color='blue', s=60, alpha=0.8, label='Predicted vs True Values')
    
    min_val = min(Y_true.min(), Y_pred.min())
    max_val = max(Y_true.max(), Y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Predictions (y=x)')
    
    plt.title('Ridge Regression: Predicted vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    output_dir = '../../outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, 'ridge_predictions_vs_true.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n拟合效果散点图已成功保存至: {save_path}")
    
    plt.close()

if __name__ == '__main__':
    try:
        data_file = '../../data/processed/data_transformed_final.xlsx'
        pd.read_excel(data_file)
    except FileNotFoundError:
        data_file = 'data_sd.xlsx'

    print("启动岭回归建模流程...")
    run_ridge_pipeline(data_file, alpha=1.0, use_full_data=True)