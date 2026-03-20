import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

def prepare_data(df, target_col):
    """
    分离特征变量 (X) 与目标变量 (Y)
    """
    X = df.drop(columns=[target_col])
    Y = df[target_col]
    return X.values, Y.values

def run_svr_pipeline(data_path, target_col, test_size=0.2, random_state=42):
    """
    执行 SVR 建模的完整流水线：加载、标准化、划分、训练、评估及可视化
    """
    df = pd.read_excel(data_path)
    X, Y = prepare_data(df, target_col)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    Y_scaled = y_scaler.fit_transform(Y.reshape(-1, 1)).ravel()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=test_size, random_state=random_state
    )

    svr_model = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1)
    svr_model.fit(X_train, Y_train)

    Y_train_pred = svr_model.predict(X_train)
    Y_test_pred = svr_model.predict(X_test)

    mse_train = mean_squared_error(Y_train, Y_train_pred)
    mse_test = mean_squared_error(Y_test, Y_test_pred)

    print("\n====== SVR 模型最优超参数 ======")
    print(f"核函数 (Kernel): {svr_model.kernel}")
    print(f"多项式次数 (Degree): {svr_model.degree}")
    print(f"正则化参数 (C): {svr_model.C}")
    print(f"容错参数 (Epsilon): {svr_model.epsilon}")

    print("\n====== 模型评估结果 (MSE) ======")
    print(f"训练集 MSE: {mse_train:.4f}")
    print(f"测试集 MSE: {mse_test:.4f}")

    plot_and_save_results(Y_train, Y_train_pred, Y_test, Y_test_pred)

def plot_and_save_results(Y_train, Y_train_pred, Y_test, Y_test_pred):
    """
    绘制标准化数据尺度的真实值与预测值拟合散点图，并保存到本地
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(Y_train, Y_train_pred, color='red', label='Train Set', s=50, alpha=0.8)
    plt.scatter(Y_test, Y_test_pred, color='blue', label='Test Set', s=50, alpha=0.8)

    all_y = np.concatenate([Y_train, Y_test])
    min_val, max_val = all_y.min(), all_y.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=3, label='Ideal Predictions (y=x)')

    plt.xlabel('True Values (Scaled)')
    plt.ylabel('Predicted Values (Scaled)')
    plt.title('SVR: Predicted vs True Values (4:1 Train-Test Split)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    output_dir = '../../outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, 'svr_predictions_vs_true.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n拟合效果图已成功保存至: {save_path}")
    
    plt.close()

if __name__ == '__main__':
    try:
        data_file = '../../data/processed/data_transformed_final.xlsx'
        temp_df = pd.read_excel(data_file)
    except FileNotFoundError:
        data_file = 'data_sd.xlsx'
        temp_df = pd.read_excel(data_file)
    
    target_column = temp_df.columns[-1]

    print(f"正在启动 SVR 建模流程...")
    print(f"目标变量: [{target_column}]")
    
    run_svr_pipeline(data_file, target_column, test_size=0.2)