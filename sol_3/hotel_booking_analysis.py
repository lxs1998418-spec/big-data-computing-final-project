#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hotel Booking Cancellation Prediction - Data Analysis Script
Analyze feature-feature relationships and feature-label relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set font for better display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """Load and explore data"""
    print("=== Loading Data ===")
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Feature classification
    categorical_features = []
    numerical_features = []
    
    for col in train_data.columns:
        if col in ['id', 'label']:
            continue
        if train_data[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    
    # Label distribution
    label_counts = train_data['label'].value_counts()
    print(f"\nLabel distribution:")
    print(f"Not Cancelled (0): {label_counts[0]:,} ({label_counts[0]/len(train_data)*100:.2f}%)")
    print(f"Cancelled (1): {label_counts[1]:,} ({label_counts[1]/len(train_data)*100:.2f}%)")
    
    return train_data, test_data, categorical_features, numerical_features

def analyze_feature_label_relationships(train_data, numerical_features, categorical_features):
    """分析特征与标签的关系"""
    print("\n" + "="*50)
    print("特征与标签关系分析")
    print("="*50)
    
    # 数值特征与标签的相关性
    numerical_corr = train_data[numerical_features + ['label']].corr()['label'].drop('label')
    numerical_corr_sorted = numerical_corr.abs().sort_values(ascending=False)
    
    print("\n数值特征与标签的相关性 (按绝对值排序):")
    for feature, corr in numerical_corr_sorted.items():
        print(f"{feature}: {corr:.4f}")
    
    # 分类特征与标签的关系 (使用卡方检验)
    print("\n分类特征与标签的关系 (卡方检验):")
    for feature in categorical_features:
        contingency_table = pd.crosstab(train_data[feature], train_data['label'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print(f"{feature}: 卡方值={chi2:.4f}, p值={p_value:.6f}")
    
    return numerical_corr_sorted

def analyze_feature_feature_relationships(train_data, numerical_features):
    """分析特征与特征的关系"""
    print("\n" + "="*50)
    print("特征与特征关系分析")
    print("="*50)
    
    # 数值特征相关性分析
    correlation_matrix = train_data[numerical_features].corr()
    
    # 找出高相关性的特征对
    print("\n高相关性特征对 (|相关系数| > 0.5):")
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                     correlation_matrix.columns[j], 
                                     corr_value))
    
    if high_corr_pairs:
        for feature1, feature2, corr in high_corr_pairs:
            print(f"{feature1} - {feature2}: {corr:.3f}")
    else:
        print("未发现高相关性特征对")
    
    return correlation_matrix

def detailed_analysis(train_data):
    """详细分析关键特征"""
    print("\n" + "="*50)
    print("关键特征详细分析")
    print("="*50)
    
    # 提前预订时间分析
    print("\n1. 提前预订时间(lead_time)分析:")
    train_data['lead_time_group'] = pd.cut(train_data['lead_time'], 
                                          bins=[0, 7, 30, 90, 365], 
                                          labels=['0-7天', '7-30天', '30-90天', '90+天'])
    
    lead_time_analysis = train_data.groupby('lead_time_group')['label'].agg(['count', 'mean'])
    lead_time_analysis.columns = ['预订数量', '取消率']
    print(lead_time_analysis)
    
    # 价格分析
    print("\n2. 房间价格(avg_price_per_room)分析:")
    train_data['price_group'] = pd.cut(train_data['avg_price_per_room'], 
                                      bins=[0, 50, 100, 150, 200, 1000], 
                                      labels=['0-50', '50-100', '100-150', '150-200', '200+'])
    
    price_analysis = train_data.groupby('price_group')['label'].agg(['count', 'mean'])
    price_analysis.columns = ['预订数量', '取消率']
    print(price_analysis)
    
    # 特殊请求分析
    print("\n3. 特殊请求数量(no_of_special_requests)分析:")
    special_requests_analysis = train_data.groupby('no_of_special_requests')['label'].agg(['count', 'mean'])
    special_requests_analysis.columns = ['预订数量', '取消率']
    print(special_requests_analysis)
    
    # 客户历史行为分析
    print("\n4. 客户历史行为分析:")
    print("重复客户 vs 取消率:")
    repeated_guest_analysis = train_data.groupby('repeated_guest')['label'].agg(['count', 'mean'])
    repeated_guest_analysis.columns = ['预订数量', '取消率']
    print(repeated_guest_analysis)
    
    print("\n历史取消次数 vs 当前取消率:")
    prev_cancellation_analysis = train_data.groupby('no_of_previous_cancellations')['label'].agg(['count', 'mean'])
    prev_cancellation_analysis.columns = ['预订数量', '取消率']
    print(prev_cancellation_analysis.head(10))

def generate_summary_report(train_data, test_data, numerical_corr_sorted, categorical_features, numerical_features):
    """生成总结报告"""
    print("\n" + "="*60)
    print("                   数据分析总结报告")
    print("="*60)
    
    label_counts = train_data['label'].value_counts()
    
    print(f"\n1. 数据集概况:")
    print(f"   - 训练样本数: {len(train_data):,}")
    print(f"   - 测试样本数: {len(test_data):,}")
    print(f"   - 特征数量: {len(train_data.columns)-2} (不包括id和label)")
    print(f"   - 数值特征: {len(numerical_features)}")
    print(f"   - 分类特征: {len(categorical_features)}")
    
    print(f"\n2. 标签分布:")
    print(f"   - 未取消 (0): {label_counts[0]:,} ({label_counts[0]/len(train_data)*100:.2f}%)")
    print(f"   - 取消 (1): {label_counts[1]:,} ({label_counts[1]/len(train_data)*100:.2f}%)")
    
    print(f"\n3. 与取消率最相关的数值特征 (前5名):")
    for i, (feature, corr) in enumerate(numerical_corr_sorted.head(5).items()):
        print(f"   {i+1}. {feature}: {corr:.4f}")
    
    print(f"\n4. 关键发现:")
    print(f"   - 提前预订时间(lead_time)与取消率呈正相关")
    print(f"   - 房间价格(avg_price_per_room)与取消率的关系需要进一步分析")
    print(f"   - 特殊请求数量(no_of_special_requests)与取消率呈负相关")
    print(f"   - 客户历史行为对当前取消行为有重要影响")
    
    print(f"\n5. 建议:")
    print(f"   - 重点关注提前预订时间较长的客户")
    print(f"   - 分析不同市场细分和房间类型的组合")
    print(f"   - 考虑客户的历史取消行为作为重要特征")
    print(f"   - 特殊请求多的客户取消率较低，可作为正面指标")
    
    print("\n" + "="*60)

def main():
    """主函数"""
    print("酒店预订取消预测 - 数据分析")
    print("="*50)
    
    # 加载和探索数据
    train_data, test_data, categorical_features, numerical_features = load_and_explore_data()
    
    # 分析特征与标签的关系
    numerical_corr_sorted = analyze_feature_label_relationships(train_data, numerical_features, categorical_features)
    
    # 分析特征与特征的关系
    correlation_matrix = analyze_feature_feature_relationships(train_data, numerical_features)
    
    # 详细分析
    detailed_analysis(train_data)
    
    # 生成总结报告
    generate_summary_report(train_data, test_data, numerical_corr_sorted, categorical_features, numerical_features)

if __name__ == "__main__":
    main()
