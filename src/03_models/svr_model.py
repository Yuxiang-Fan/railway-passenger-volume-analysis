import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def fit_svr_model(X, y, test_size=0.2):
    # 针对时间序列/宏观经济数据，关闭打乱顺序，严格按照时间先后划分，防止未来数据泄露
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # 核心修正：必须先划分数据集，再单独对训练集 fit，严格隔离测试集信息
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # 实例化 SVR，参数假设基于前期超参数搜索得出
    model = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train_scaled)

    pred_train_scaled = model.predict(X_train_scaled)
    pred_test_scaled = model.predict(X_test_scaled)

    mse_train = mean_squared_error(y_train_scaled, pred_train_scaled)
    mse_test = mean_squared_error(y_test_scaled, pred_test_scaled)
    
    print("--- SVR Model Performance ---")
    print(f">> Train MSE: {mse_train:.4e}")
    print(f">> Test MSE:  {mse_test:.4e}")
    print(f">> Hyperparameters: Kernel={model.kernel}, Degree={model.degree}, C={model.C}, Epsilon={model.epsilon}")

    return model, (y_train_scaled, pred_train_scaled, y_test_scaled, pred_test_scaled)

def plot_svr_fit(y_train, pred_train, y_test, pred_test):
    # 绘制标准化尺度下的拟合对角线散点图
    plt.figure(figsize=(8, 8))

    plt.scatter(y_train, pred_train, color='#d62728', label='Train Set', s=50, alpha=0.7)
    plt.scatter(y_test, pred_test, color='#1f77b4', label='Test Set', s=50, alpha=0.7)

    all_y = np.concatenate([y_train, y_test])
    axis_min, axis_max = all_y.min(), all_y.max()
    plt.plot([axis_min, axis_max], [axis_min, axis_max], color='black', linestyle='--', linewidth=2, label='Ideal Fit (y=x)')

    plt.xlabel('Actual Target (Scaled)')
    plt.ylabel('Predicted Target (Scaled)')
    plt.title('SVR: Predicted vs Actual Target')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    out_dir = 'outputs/figures'
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, 'svr_fit_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n>> 拟合散点图已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    data_file = 'data/processed/dataset_ready.xlsx'
    
    if os.path.exists(data_file):
        df = pd.read_excel(data_file)
        
        # 剥离时间索引，提取特征与目标变量
        X_data = df.iloc[:, 1:-1]
        y_data = df.iloc[:, -1]
        
        trained_model, plot_data = fit_svr_model(X_data, y_data, test_size=0.2)
        plot_svr_fit(*plot_data)
    else:
        print(f">> [提示] 未找到 {data_file}。")
