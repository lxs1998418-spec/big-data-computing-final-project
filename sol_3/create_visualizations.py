#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒店预订取消预测 - 可视化分析
创建关键分析结果的可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def create_visualizations():
    """创建可视化图表"""
    # 读取数据
    train_data = pd.read_csv('train.csv')
    
    # 特征分类
    categorical_features = []
    numerical_features = []
    
    for col in train_data.columns:
        if col in ['id', 'label']:
            continue
        if train_data[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    # 创建图表
    fig = plt.figure(figsize=(20, 24))
    
    # 1. 标签分布饼图
    plt.subplot(4, 3, 1)
    label_counts = train_data['label'].value_counts()
    colors = ['lightblue', 'lightcoral']
    labels = ['未取消 (0)', '取消 (1)']
    plt.pie(label_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('标签分布', fontsize=14, fontweight='bold')
    
    # 2. 数值特征与标签相关性
    plt.subplot(4, 3, 2)
    numerical_corr = train_data[numerical_features + ['label']].corr()['label'].drop('label')
    numerical_corr_sorted = numerical_corr.abs().sort_values(ascending=False)
    
    # 取前8个最相关的特征
    top_features = numerical_corr_sorted.head(8)
    colors = ['red' if train_data[numerical_features + ['label']].corr()['label'][feat] > 0 else 'blue' 
              for feat in top_features.index]
    
    bars = plt.bar(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
    plt.title('数值特征与标签相关性 (前8名)', fontsize=14, fontweight='bold')
    plt.xlabel('特征')
    plt.ylabel('相关系数绝对值')
    plt.xticks(range(len(top_features)), top_features.index, rotation=45, ha='right')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. 提前预订时间与取消率
    plt.subplot(4, 3, 3)
    train_data['lead_time_group'] = pd.cut(train_data['lead_time'], 
                                          bins=[0, 7, 30, 90, 365], 
                                          labels=['0-7天', '7-30天', '30-90天', '90+天'])
    
    lead_time_analysis = train_data.groupby('lead_time_group')['label'].agg(['count', 'mean'])
    lead_time_analysis.columns = ['预订数量', '取消率']
    
    bars = plt.bar(range(len(lead_time_analysis)), lead_time_analysis['取消率'], 
                   color='salmon', alpha=0.7)
    plt.title('提前预订时间与取消率', fontsize=14, fontweight='bold')
    plt.xlabel('提前预订时间')
    plt.ylabel('取消率')
    plt.xticks(range(len(lead_time_analysis)), lead_time_analysis.index, rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. 房间价格与取消率
    plt.subplot(4, 3, 4)
    train_data['price_group'] = pd.cut(train_data['avg_price_per_room'], 
                                      bins=[0, 50, 100, 150, 200, 1000], 
                                      labels=['0-50', '50-100', '100-150', '150-200', '200+'])
    
    price_analysis = train_data.groupby('price_group')['label'].agg(['count', 'mean'])
    price_analysis.columns = ['预订数量', '取消率']
    
    bars = plt.bar(range(len(price_analysis)), price_analysis['取消率'], 
                   color='lightgreen', alpha=0.7)
    plt.title('房间价格与取消率', fontsize=14, fontweight='bold')
    plt.xlabel('价格区间')
    plt.ylabel('取消率')
    plt.xticks(range(len(price_analysis)), price_analysis.index, rotation=45)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. 特殊请求数量与取消率
    plt.subplot(4, 3, 5)
    special_requests_analysis = train_data.groupby('no_of_special_requests')['label'].agg(['count', 'mean'])
    special_requests_analysis.columns = ['预订数量', '取消率']
    
    bars = plt.bar(range(len(special_requests_analysis)), special_requests_analysis['取消率'], 
                   color='purple', alpha=0.7)
    plt.title('特殊请求数量与取消率', fontsize=14, fontweight='bold')
    plt.xlabel('特殊请求数量')
    plt.ylabel('取消率')
    plt.xticks(range(len(special_requests_analysis)), special_requests_analysis.index)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 6. 重复客户与取消率
    plt.subplot(4, 3, 6)
    repeated_guest_analysis = train_data.groupby('repeated_guest')['label'].agg(['count', 'mean'])
    repeated_guest_analysis.columns = ['预订数量', '取消率']
    
    bars = plt.bar(range(len(repeated_guest_analysis)), repeated_guest_analysis['取消率'], 
                   color='orange', alpha=0.7)
    plt.title('重复客户与取消率', fontsize=14, fontweight='bold')
    plt.xlabel('重复客户 (0=否, 1=是)')
    plt.ylabel('取消率')
    plt.xticks(range(len(repeated_guest_analysis)), ['否', '是'])
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 7. 市场细分类型与取消率
    plt.subplot(4, 3, 7)
    market_segment_analysis = train_data.groupby('market_segment_type')['label'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(market_segment_analysis)), market_segment_analysis.values, 
                   color='skyblue', alpha=0.7)
    plt.title('市场细分类型与取消率', fontsize=14, fontweight='bold')
    plt.xlabel('市场细分类型')
    plt.ylabel('取消率')
    plt.xticks(range(len(market_segment_analysis)), market_segment_analysis.index, rotation=45, ha='right')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 8. 房间类型与取消率
    plt.subplot(4, 3, 8)
    room_type_analysis = train_data.groupby('room_type_reserved')['label'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(room_type_analysis)), room_type_analysis.values, 
                   color='lightcoral', alpha=0.7)
    plt.title('房间类型与取消率', fontsize=14, fontweight='bold')
    plt.xlabel('房间类型')
    plt.ylabel('取消率')
    plt.xticks(range(len(room_type_analysis)), room_type_analysis.index, rotation=45, ha='right')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 9. 餐食计划与取消率
    plt.subplot(4, 3, 9)
    meal_plan_analysis = train_data.groupby('type_of_meal_plan')['label'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(meal_plan_analysis)), meal_plan_analysis.values, 
                   color='lightgreen', alpha=0.7)
    plt.title('餐食计划与取消率', fontsize=14, fontweight='bold')
    plt.xlabel('餐食计划')
    plt.ylabel('取消率')
    plt.xticks(range(len(meal_plan_analysis)), meal_plan_analysis.index, rotation=45, ha='right')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 10. 数值特征相关性热力图
    plt.subplot(4, 3, 10)
    correlation_matrix = train_data[numerical_features].corr()
    
    # 只显示前8个最相关的特征
    top_numerical_features = numerical_corr_sorted.head(8).index.tolist()
    corr_subset = correlation_matrix.loc[top_numerical_features, top_numerical_features]
    
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('数值特征相关性矩阵 (前8名)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 11. 历史取消次数与当前取消率
    plt.subplot(4, 3, 11)
    prev_cancellation_analysis = train_data.groupby('no_of_previous_cancellations')['label'].agg(['count', 'mean'])
    prev_cancellation_analysis.columns = ['预订数量', '取消率']
    
    # 只显示前8个最常见的值
    top_prev_cancellations = prev_cancellation_analysis.head(8)
    
    bars = plt.bar(range(len(top_prev_cancellations)), top_prev_cancellations['取消率'], 
                   color='red', alpha=0.7)
    plt.title('历史取消次数与当前取消率', fontsize=14, fontweight='bold')
    plt.xlabel('历史取消次数')
    plt.ylabel('当前取消率')
    plt.xticks(range(len(top_prev_cancellations)), top_prev_cancellations.index)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 12. 历史未取消预订次数与当前取消率
    plt.subplot(4, 3, 12)
    prev_not_cancelled_analysis = train_data.groupby('no_of_previous_bookings_not_canceled')['label'].agg(['count', 'mean'])
    prev_not_cancelled_analysis.columns = ['预订数量', '取消率']
    
    # 只显示前8个最常见的值
    top_prev_not_cancelled = prev_not_cancelled_analysis.head(8)
    
    bars = plt.bar(range(len(top_prev_not_cancelled)), top_prev_not_cancelled['取消率'], 
                   color='green', alpha=0.7)
    plt.title('历史未取消预订次数与当前取消率', fontsize=14, fontweight='bold')
    plt.xlabel('历史未取消预订次数')
    plt.ylabel('当前取消率')
    plt.xticks(range(len(top_prev_not_cancelled)), top_prev_not_cancelled.index)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('hotel_booking_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可视化图表已保存为 'hotel_booking_analysis_visualization.png'")

if __name__ == "__main__":
    create_visualizations()
