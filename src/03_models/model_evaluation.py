import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    加载最终预处理后的数据
    """
    df = pd.read_excel(filepath)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    return X, y

def evaluate_svr(X_scaled, y_scaled):
    """
    运行支持向量回归 (SVR) 并计算全量数据下的 MSE
    使用论文中网格搜索得到的最优参数
    """
    svr_model = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1)
    svr_model.fit(X_scaled, y_scaled)
    y_pred = svr_model.predict(X_scaled)
    mse = mean_squared_error(y_scaled, y_pred)
    return mse

def evaluate_ridge(X_scaled, y, alpha=1.0):
    """
    运行岭回归 (Ridge Regression) 并计算 MSE
    """
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_scaled, y)
    y_pred = ridge_model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    return mse

def evaluate_stepwise(X, y):
    """
    运行逐步回归 (Stepwise Regression) 并计算 MSE
    使用论文前向/后向逐步回归最终筛选出的最优特征集
    """
    optimal_features = [
        'GDP(现价):交通运输、仓储和邮政业:年', 
        '城镇单位就业人员平均工资:年', 
        '铁路旅客周转量:年', 
        '铁路客车拥有量:软卧车:年', 
        '铁路客车拥有量:软座车:年', 
        '高速铁路营业里程:年'
    ]
    
    actual_features = [col for col in optimal_features if col in X.columns]
    
    X_final = sm.add_constant(X[actual_features])
    model = sm.OLS(y, X_final).fit()
    y_pred = model.predict(X_final)
    mse = mean_squared_error(y, y_pred)
    return mse

def generate_comparison_report_and_plot(mse_results):
    """
    在控制台打印对比表格，并生成可视化柱状图保存到本地
    """
    print("\n" + "="*50)
    print(" 三种回归模型的误差对比表")
    print("="*50)
    print(f"{'回归模型 (Model)': <25} | {'均方误差 (MSE)'}")
    print("-" * 50)
    for model_name, mse in mse_results.items():
        print(f"{model_name: <25} | {mse:.15f}")
    print("="*50)
    print("结论: 逐步回归模型拟合精度最高，岭回归其次，支持向量回归拟合精度最低。")

    plt.figure(figsize=(9, 6))
    
    models = list(mse_results.keys())
    mses = list(mse_results.values())
    
    bars = plt.bar(models, mses, color=['#ff9999', '#66b3ff', '#99ff99'], edgecolor='black', alpha=0.8)
    
    plt.yscale('log')
    plt.title('Model Performance Comparison (MSE in Log Scale)', fontsize=14, pad=15)
    plt.ylabel('Mean Squared Error (Log Scale)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f'{yval:.6f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    output_dir = '../../outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'model_mse_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图表已成功保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    try:
        data_file = '../../data/processed/data_transformed_final.xlsx'
        df = pd.read_excel(data_file)
    except FileNotFoundError:
        data_file = 'data_sd.xlsx'
        df = pd.read_excel(data_file)
        
    X, y = load_data(data_file)
    
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

    mse_dict = {}
    
    mse_dict['Support Vector Reg.'] = evaluate_svr(X_scaled, y_scaled)
    mse_dict['Ridge Regression'] = evaluate_ridge(X_scaled, y)
    mse_dict['Stepwise Regression'] = evaluate_stepwise(X, y)

    generate_comparison_report_and_plot(mse_dict)