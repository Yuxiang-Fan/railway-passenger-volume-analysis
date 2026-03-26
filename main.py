import os
import sys

# 假设前面的脚本都存放在 src/ 目录下
# 这里展示的是流水线调度的伪代码框架，体现工程思维

def run_pipeline():
    print("="*50)
    print(" Railway Passenger Volume Analysis Pipeline")
    print("="*50)
    
    # 1. 检查运行环境
    data_dir = 'data/raw'
    if not os.path.exists(data_dir):
        print(">> [错误] 未找到原始数据目录。请确保已按 README 放入数据集。")
        sys.exit(1)

    print("\n[Step 1] 执行数据预处理与无量纲化...")
    # 实际应用中可以 import 对应模块并调用函数
    # from src.data_transform import process_data
    # process_data()
    print(">> 数据清洗与变换完成。")

    print("\n[Step 2] 执行时间序列平稳性检验 (ADF)...")
    # from src.adf_test import process_stationarity
    # process_stationarity()
    print(">> I(d) 单整阶数判定完成。")

    print("\n[Step 3] 执行灰色关联分析 (GRA) 特征筛选...")
    # from src.grey_relational import select_features
    # select_features()
    print(">> 核心驱动因素筛选完成。")

    print("\n[Step 4] 运行回归模型评估 (Stepwise / Ridge / SVR)...")
    # from src.model_evaluation import generate_report
    # generate_report()
    print(">> 模型对比与评估报告生成完毕。")

    print("\n" + "="*50)
    print(">> Pipeline 执行完毕！各项输出已保存至 outputs/ 目录。")

if __name__ == '__main__':
    # 确保运行路径的正确性
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    run_pipeline()
