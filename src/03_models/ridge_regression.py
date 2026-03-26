import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def fit_ridge_model(X, y, alpha=1.0):
    # 岭回归核心拟合逻辑 (L2 正则化)
    # 增加 StandardScaler 保底，防止上游未进行无量纲化导致惩罚项权重失衡
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    
    y_pred = model.predict(X_scaled)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"--- Ridge Regression Summary (Alpha={alpha}) ---")
    print(f">> MSE: {mse:.4e}")
    print(f">> R-squared: {r2:.4f}\n")
    
    print(">> 特征系数权重:")
    for feat, coef in zip(X.columns, model.coef_):
        print(f" - {feat}: {coef:.6f}")
        
    print(f"\n>> 截距项 (Intercept): {model.intercept_:.6f}")
    
    return model, y_pred

def plot_ridge_fit(y_true, y_pred):
    # 绘制真实值与预测值的对角线拟合散点图
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_true, y_pred, color='#1f77b4', s=50, alpha=0.7, label='Predicted vs Actual')
    
    # 绘制 y=x 理想参考线
    axis_min = min(y_true.min(), y_pred.min())
    axis_max = max(y_true.max(), y_pred.max())
    plt.plot([axis_min, axis_max], [axis_min, axis_max], color='#d62728', linestyle='--', linewidth=2, label='Ideal Fit (y=x)')
    
    plt.title('Ridge Regression: Fit Performance')
    plt.xlabel('Actual Target Value')
    plt.ylabel('Predicted Target Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    out_dir = 'outputs/figures'
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, 'ridge_fit_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f">> 拟合散点图已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    data_file = 'data/processed/dataset_ready.xlsx'
    
    if os.path.exists(data_file):
        df = pd.read_excel(data_file)
        
        # 假设第一列为时间索引，剥离后送入模型
        time_col = df.columns[0]
        X_data = df.iloc[:, 1:-1]
        y_data = df.iloc[:, -1]
        
        # 执行岭回归，alpha 值可根据实际超参数调优结果替换
        model, predictions = fit_ridge_model(X_data, y_data, alpha=1.0)
        plot_ridge_fit(y_data, predictions)
    else:
        print(f">> [提示] 未找到 {data_file}。")
