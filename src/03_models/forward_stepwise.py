import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """
    加载数据并分离特征与目标变量
    """
    df = pd.read_excel(filepath)
    X = df.iloc[:, 1:-1]
    Y = df.iloc[:, -1]
    return X, Y

def forward_stepwise_selection(X, y):
    """
    真正的前向逐步回归算法 (贪婪算法)
    以 AIC (赤池信息准则) 为评估指标，逐步将特征加入模型
    """
    initial_features = X.columns.tolist()
    best_features = []
    best_aic = float('inf')
    
    print("\n====== 开始执行前向逐步回归特征筛选 ======")
    
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype=float)
        
        step_aic_dict = {}
        
        for new_column in remaining_features:
            model_features = best_features + [new_column]
            X_subset = sm.add_constant(X[model_features])
            
            model = sm.OLS(y, X_subset).fit()
            step_aic_dict[new_column] = model.aic
            
        best_step_feature = min(step_aic_dict, key=step_aic_dict.get)
        best_step_aic = step_aic_dict[best_step_feature]
        
        if best_step_aic < best_aic:
            best_features.append(best_step_feature)
            best_aic = best_step_aic
            print(f" [+] 选入特征: {best_step_feature: <25} | 当前模型 AIC: {best_aic:.4f}")
        else:
            print(f"\n [!] 停止搜索：加入其余任何特征均无法进一步降低 AIC。")
            break
            
    return best_features

def evaluate_and_plot_model(X, y, selected_features):
    """
    基于筛选出的最优特征子集，输出模型报告并绘制拟合图
    """
    print("\n====== 最终最优模型摘要 ======")
    print(f"最终入选的特征集: {selected_features}")
    
    X_final = sm.add_constant(X[selected_features])
    best_model = sm.OLS(y, X_final).fit()
    
    print(best_model.summary())
    
    print("\n====== 模型参数 ======")
    print(best_model.params)
    
    y_pred = best_model.predict(X_final)
    mse = mean_squared_error(y, y_pred)
    print(f"\n====== 均方误差 (MSE) ======")
    print(f"Mean Squared Error (MSE): {mse:.15f}")
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(y.values, label='Actual (真实值)', marker='o', linestyle='-', alpha=0.8)
    plt.plot(y_pred.values, label='Predicted (预测值)', marker='^', linestyle='--', alpha=0.8)
    
    plt.title('Forward Stepwise Regression: Actual vs Predicted')
    plt.xlabel('Observation (样本序号)')
    plt.ylabel('Value (客运量)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_dir = '../../outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'forward_stepwise_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已成功保存至: {save_path}")
    
    plt.close()

if __name__ == '__main__':
    try:
        data_file = '../../data/processed/data_transformed_final.xlsx'
        pd.read_excel(data_file)
    except FileNotFoundError:
        data_file = 'data_sd.xlsx'
        
    print("启动前向逐步回归建模流程...")
    
    X_data, y_data = load_and_prepare_data(data_file)
    
    optimal_features = forward_stepwise_selection(X_data, y_data)
    
    evaluate_and_plot_model(X_data, y_data, optimal_features)