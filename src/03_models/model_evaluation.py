import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

def eval_svr(X, y):
    # 使用 Poly 核的 SVR，参数为网格搜索最优解
    model = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1)
    model.fit(X, y)
    pred = model.predict(X)
    return mean_squared_error(y, pred), r2_score(y, pred)

def eval_ridge(X, y, alpha=1.0):
    # 岭回归 (L2 正则化)
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    pred = model.predict(X)
    return mean_squared_error(y, pred), r2_score(y, pred)

def eval_stepwise(X, y, features):
    # OLS 回归 (基于逐步回归筛选的特征子集)
    valid_features = [col for col in features if col in X.columns]
    X_final = sm.add_constant(X[valid_features])
    
    model = sm.OLS(y, X_final).fit()
    pred = model.predict(X_final)
    return mean_squared_error(y, pred), r2_score(y, pred)

def generate_report(metrics_dict):
    # 生成性能对比报告与可视化柱状图
    print("\n--- Regression Models Performance ---")
    print(f"{'Model': <20} | {'MSE': <15} | {'R-squared'}")
    print("-" * 55)
    
    for name, metrics in metrics_dict.items():
        print(f"{name: <20} | {metrics['mse']:.6e} | {metrics['r2']:.4f}")
        
    plt.figure(figsize=(9, 6))
    
    models = list(metrics_dict.keys())
    mses = [m['mse'] for m in metrics_dict.values()]
    
    bars = plt.bar(models, mses, color=['#ff9999', '#66b3ff', '#99ff99'], edgecolor='black', alpha=0.8)
    
    plt.yscale('log')
    plt.title('Model Performance Comparison (MSE in Log Scale)')
    plt.ylabel('Mean Squared Error (Log Scale)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f'{yval:.2e}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    out_dir = 'outputs/figures'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'model_mse_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n>> 对比图表已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    data_file = 'data/processed/dataset_ready.xlsx'
    
    if os.path.exists(data_file):
        df = pd.read_excel(data_file)
        # 假设数据在送入此脚本前，X 和 y 已在流水线中统一完成了无量纲化处理
        X_data = df.iloc[:, 1:-1]
        y_data = df.iloc[:, -1]
        
        # 模拟由 forward/backward stepwise 脚本传递过来的最优特征集
        optimal_subset = [
            'GDP(现价):交通运输、仓储和邮政业:年', 
            '城镇单位就业人员平均工资:年', 
            '铁路旅客周转量:年', 
            '铁路客车拥有量:软卧车:年', 
            '铁路客车拥有量:软座车:年', 
            '高速铁路营业里程:年'
        ]

        results = {}
        
        mse, r2 = eval_svr(X_data, y_data)
        results['SVR (Poly)'] = {'mse': mse, 'r2': r2}
        
        mse, r2 = eval_ridge(X_data, y_data)
        results['Ridge Reg'] = {'mse': mse, 'r2': r2}
        
        mse, r2 = eval_stepwise(X_data, y_data, optimal_subset)
        results['Stepwise OLS'] = {'mse': mse, 'r2': r2}

        generate_report(results)
    else:
        print(f">> [提示] 未找到 {data_file}。")
