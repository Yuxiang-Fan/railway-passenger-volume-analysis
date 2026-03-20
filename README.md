# Railway Passenger Volume Factor Analysis
# 铁路客运量影响因素统计分析

This repository contains the source code and documentation for a study on the influencing factors of national railway passenger volume, originally developed for the 10th National Undergraduate Statistical Modeling Competition (2024).

本项目包含针对全国铁路客运量影响因素研究的源代码及文档，最初为第十届全国大学生统计建模大赛（2024）开发。

---

## English Version

### ⚠️ Dataset Notice
**The dataset is NOT provided in this repository** due to competition regulations and data sensitivity. Users are required to independently obtain the relevant datasets from the official sources of the **10th National Undergraduate Statistical Modeling Competition (2024)**.

### Project Overview
The study aims to identify key drivers of railway passenger volume in China by analyzing data from 2008 to 2021. Given the impact of digital transformation and infrastructure investment, understanding these variables is essential for future transportation planning.

### Methodology
1. **Data Preprocessing**: Newton interpolation was utilized to address missing data points. Features underwent de-dimensionalization and logarithmic transformation to mitigate heteroscedasticity and stabilize variance.
2. **Feature Engineering**: 
    * **Grey Relational Analysis (GRA)**: Used to rank potential indicators such as GDP, urban wages, and railway mileage based on their correlation with passenger volume.
    * **Statistical Testing**: ADF tests were conducted to check for stationarity. Granger causality tests were explored but faced limitations due to the small sample size.
3. **Modeling**: The project compares three regression techniques: Support Vector Regression (SVR), Ridge Regression, and Stepwise Regression.

### Key Findings
* **Stepwise Regression** achieved the highest accuracy with a Mean Squared Error (MSE) of approximately **0.000183**.
* Key significant factors include average wages of urban employees and railway passenger turnover.
* The analysis suggests that increasing high-end service capacity (e.g., soft sleepers) correlates positively with passenger volume.

---

## 中文版

### ⚠️ 数据集说明
**本仓库不提供原始数据集**。受限于竞赛规定及数据敏感性，用户需自行从**第十届全国大学生统计建模大赛（2024）**官方渠道获取相关数据。

### 项目简介
本项目旨在通过分析 2008 年至 2021 年的数据，识别影响中国铁路客运量的核心驱动因素。考虑到数字化转型和基础设施投资的影响，理解这些变量对未来的交通规划至关重要。

### 研究方法
1. **数据预处理**：采用牛顿插值法处理缺失值。对特征进行了去量纲化和对数变换，以减弱异方差性并增强数据稳定性。
2. **特征工程**：
    * **灰色关联分析 (GRA)**：根据指标（如 GDP、城镇工资、铁路里程）与客运量的关联程度进行排序。
    * **统计检验**：通过 ADF 检验验证平稳性。对格兰杰因果检验进行了探索，但受限于样本量较小，存在一定的局限性。
3. **模型建立**：项目对比了三种回归技术：支持向量回归 (SVR)、岭回归和逐步回归。

### 核心结论
* **逐步回归模型**表现最佳，均方误差 (MSE) 约为 **0.000183**。
* 城镇单位就业人员平均工资和铁路旅客周转量被确定为核心显著影响因素。
* 分析表明，增加高品质服务（如软卧车）的供给与客运量提升呈正相关。

---

## 📁 Repository Structure / 仓库结构

```text
.
├── data/raw/              # Place your obtained datasets here / 请将获取的数据放入此处
├── docs/                  # Research paper (PDF) / 研究论文
├── src/                   # Source code modules / 源代码模块
└── main.py                # Execution entry / 执行入口
```
