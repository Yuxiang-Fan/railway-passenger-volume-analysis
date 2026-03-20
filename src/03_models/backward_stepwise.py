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

def backward_stepwise_selection(X, y):
    """
    真正的后向逐步回归算法
    从包含所有特征的全模型开始，每次剔除一个最不重要的特征，直到 AIC 无法进一步降低
    """
    current_features = X.columns.tolist()
    
    X_initial = sm.add_constant(X[current_features])
    initial_model = sm.OLS(y, X_initial).fit()
    best_aic = initial_model.aic
    
    print("\n====== 开始执行后向逐步回归特征剔除 ======")
    print(f" [Start] 初始全量特征模型 AIC: {best_aic:.4f}")
    
    while len(current_features) > 0:
        step_aic_dict = {}
        
        for feature_to_remove in current_features:
            reduced_features = list(set(current_features) - {feature_to_remove})
            
            if not reduced_features:
                X_subset = sm.add_constant(pd.DataFrame(index=X.index))
            else:
                X_subset = sm.add_constant(X[reduced_features])
            
            model = sm.OLS(y, X_subset).fit()
            step_aic_dict[feature_to_remove] = model.aic
            
        best_feature_to_remove = min(step_aic_dict, key=step_aic_dict.get)
        best_step_aic = step_aic_dict[best_feature_to_remove]
        
        if best_step_aic < best_aic:
            current_features.remove(best_feature_to_remove)
            best_aic = best_step_aic
            print(f" [-] 成功剔除: {best_feature_to_remove: <25} | 当前模型 AIC: {best_aic:.4f}")
        else:
            print(f"\n [!] 停止搜索：继续剔除任何现有特征均会导致 AIC 上升。")
            break
            
    return current_features

def evaluate_and_plot_model(X, y, selected_features):
    """
    基于筛选出的最优特征子集，输出模型报告并绘制拟合图
    """
    print("\n====== 最终最优模型摘要 ======")
    print(f"最终保留的核心特征集: {selected_features}")
    
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
    
    plt.title('Backward Stepwise Regression: Actual vs Predicted')
    plt.xlabel('Observation (样本序号)')
    plt.ylabel('Value (客运量)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_dir = '../../outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'backward_stepwise_predictions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已成功保存至: {save_path}")
    
    plt.close()

if __name__ == '__main__':
    try:
        data_file = '../../data/processed/data_transformed_final.xlsx'
        pd.read_excel(data_file)
    except FileNotFoundError:
        data_file = 'data_sd.xlsx'
        
    print("启动后向逐步回归建模流程...")
    
    X_data, y_data = load_and_prepare_data(data_file)
    
    optimal_features = backward_stepwise_selection(X_data, y_data)
    
    evaluate_and_plot_model(X_data, y_data, optimal_features)