#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征相关性热力图分析
基于 house_predict_pytorch_my.py 中的特征工程，对处理后的特征进行相关性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def preprocess_data(train_df, test_df):
    """
    与 house_predict_pytorch_my.py 相同的预处理函数
    """
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    print("开始数据预处理...")
    
    # 处理分类变量
    categorical_cols = ['country', 'property_type', 'furnishing_status']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        combined_data = pd.concat([train_processed[col], test_processed[col]])
        le.fit(combined_data)
        train_processed[col] = le.transform(train_processed[col])
        test_processed[col] = le.transform(test_processed[col])
        label_encoders[col] = le

    # 特征工程
    
    # 可负担性比率
    train_processed['affordability_ratio'] = train_processed['customer_salary'] / (train_processed['price'] + 1)
    test_processed['affordability_ratio'] = test_processed['customer_salary'] / (test_processed['price'] + 1)

    # 贷款价值比
    train_processed['loan_to_value'] = train_processed['loan_amount'] / (train_processed['price'] + 1)
    test_processed['loan_to_value'] = test_processed['loan_amount'] / (test_processed['price'] + 1)

    # 房产年龄
    current_year = 2025
    train_processed['property_age'] = current_year - train_processed['constructed_year']
    test_processed['property_age'] = current_year - test_processed['constructed_year']

    # 支付能力
    train_processed['payment_capacity'] = train_processed['customer_salary'] - train_processed['monthly_expenses']
    test_processed['payment_capacity'] = test_processed['customer_salary'] - test_processed['monthly_expenses']

    # 首付比率
    train_processed['down_payment_ratio'] = train_processed['down_payment'] / (train_processed['price'] + 1)
    test_processed['down_payment_ratio'] = test_processed['down_payment'] / (test_processed['price'] + 1)

    # 风险评分
    train_processed['risk_score'] = train_processed['crime_cases_reported'] + train_processed['legal_cases_on_property']
    test_processed['risk_score'] = test_processed['crime_cases_reported'] + test_processed['legal_cases_on_property']

    # 质量评分
    train_processed['quality_score'] = train_processed['satisfaction_score'] + train_processed['neighbourhood_rating'] + train_processed['connectivity_score']
    test_processed['quality_score'] = test_processed['satisfaction_score'] + test_processed['neighbourhood_rating'] + test_processed['connectivity_score']

    print("数据预处理完成!")
    return train_processed, test_processed, label_encoders


def plot_correlation_heatmap(df, title="特征相关性热力图", figsize=(16, 14), save_path="feature_correlation_heatmap.png", include_label=True):
    """
    绘制特征相关性热力图
    
    Args:
        df: 包含特征的DataFrame
        title: 图表标题
        figsize: 图表大小
        save_path: 保存路径
        include_label: 是否包含标签（label）在相关性分析中
    """
    # 选择数值特征（排除id，根据include_label决定是否包含label）
    if include_label and 'label' in df.columns:
        feature_cols = [col for col in df.columns if col not in ['id']]
    else:
        feature_cols = [col for col in df.columns if col not in ['id', 'label']]
    feature_df = df[feature_cols]
    
    # 计算相关性矩阵
    print(f"\n计算 {len(feature_cols)} 个特征的相关性矩阵（包含标签: {include_label and 'label' in df.columns}）...")
    correlation_matrix = feature_df.corr()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)  # 只显示下三角，避免重复
    
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('特征', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)
    
    # 旋转标签以便更好地阅读
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存到: {save_path}")
    plt.close()


def plot_correlation_with_target(df, target_col='label', top_n=20, figsize=(12, 10), save_path="feature_target_correlation.png"):
    """
    绘制与目标变量的相关性条形图
    
    Args:
        df: 包含特征和目标变量的DataFrame
        target_col: 目标变量列名
        top_n: 显示前N个最相关的特征
        figsize: 图表大小
        save_path: 保存路径
    """
    # 选择数值特征
    feature_cols = [col for col in df.columns if col not in ['id', target_col]]
    feature_df = df[feature_cols + [target_col]]
    
    # 计算与目标变量的相关性
    correlations = feature_df.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    
    # 选择前N个
    top_correlations = correlations.head(top_n)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制条形图
    colors = ['red' if x < 0 else 'blue' for x in top_correlations.values]
    bars = ax.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7)
    
    # 设置标签
    ax.set_yticks(range(len(top_correlations)))
    ax.set_yticklabels(top_correlations.index)
    ax.set_xlabel('与目标变量的相关系数', fontsize=12)
    ax.set_title(f'与目标变量相关性最高的前{top_n}个特征', fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (idx, val) in enumerate(top_correlations.items()):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"目标变量相关性图已保存到: {save_path}")
    plt.close()
    
    return correlations


def analyze_high_correlations(correlation_matrix, threshold=0.7):
    """
    分析高相关性特征对
    
    Args:
        correlation_matrix: 相关性矩阵
        threshold: 相关性阈值
    """
    print(f"\n分析高相关性特征对 (阈值 >= {threshold}):")
    print("=" * 60)
    
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
        print(high_corr_df.to_string(index=False))
        print(f"\n共找到 {len(high_corr_pairs)} 对高相关性特征")
    else:
        print(f"未找到相关性 >= {threshold} 的特征对")
    
    return high_corr_pairs


def main():
    print("=" * 60)
    print("特征相关性热力图分析")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    
    # 2. 数据预处理和特征工程
    print("\n2. 数据预处理和特征工程...")
    train_processed, test_processed, encoders = preprocess_data(train_df, test_df)
    
    # 合并训练和测试数据以获得完整的特征集（用于相关性分析）
    # 注意：这里我们主要使用训练数据进行相关性分析
    print("\n3. 准备特征数据...")
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'label']]
    print(f"特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols}")
    
    # 4. 绘制完整特征相关性热力图（包含标签）
    print("\n4. 绘制特征相关性热力图（包含标签）...")
    plot_correlation_heatmap(
        train_processed,
        title="特征工程后的特征相关性热力图（包含标签）",
        figsize=(20, 18),
        save_path="feature_correlation_heatmap.png",
        include_label=True
    )
    
    # 5. 分析与目标变量的相关性
    print("\n5. 分析与目标变量的相关性...")
    target_correlations = plot_correlation_with_target(
        train_processed,
        target_col='label',
        top_n=20,
        figsize=(12, 10),
        save_path="feature_target_correlation.png"
    )
    
    print("\n与目标变量相关性最高的10个特征:")
    print(target_correlations.head(10).to_string())
    
    # 6. 分析高相关性特征对（包含标签）
    feature_df = train_processed[[col for col in train_processed.columns if col not in ['id']]]
    correlation_matrix = feature_df.corr()
    high_corr_pairs = analyze_high_correlations(correlation_matrix, threshold=0.7)
    
    # 特别分析标签与其他特征的高相关性
    if 'label' in correlation_matrix.columns:
        print("\n6.1 与标签相关性最高的特征:")
        print("=" * 60)
        label_correlations = correlation_matrix['label'].drop('label').sort_values(key=abs, ascending=False)
        print(label_correlations.to_string())
    
    # 7. 特征统计信息
    print("\n7. 特征统计信息:")
    print("=" * 60)
    # 用于统计的特征列表（不包含label）
    stats_feature_cols = [col for col in train_processed.columns if col not in ['id', 'label']]
    print(f"总特征数（不含标签）: {len(stats_feature_cols)}")
    print(f"原始特征数: {len([col for col in train_df.columns if col not in ['id', 'label']])}")
    print(f"新增特征数: {len(stats_feature_cols) - len([col for col in train_df.columns if col not in ['id', 'label']])}")
    
    print("\n新增特征列表:")
    original_cols = set([col for col in train_df.columns if col not in ['id', 'label']])
    new_features = [col for col in stats_feature_cols if col not in original_cols]
    for feat in new_features:
        print(f"  - {feat}")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

