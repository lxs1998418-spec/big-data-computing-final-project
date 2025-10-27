#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hotel Booking Cancellation Prediction - Visualization Analysis
Create visualizations for key analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set font for better display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def create_visualizations():
    """Create visualization charts"""
    # Load data
    train_data = pd.read_csv('train.csv')
    
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
    
    # Create charts
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Label distribution pie chart
    plt.subplot(4, 3, 1)
    label_counts = train_data['label'].value_counts()
    colors = ['lightblue', 'lightcoral']
    labels = ['Not Cancelled (0)', 'Cancelled (1)']
    plt.pie(label_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Label Distribution', fontsize=14, fontweight='bold')
    
    # 2. Numerical features correlation with label
    plt.subplot(4, 3, 2)
    numerical_corr = train_data[numerical_features + ['label']].corr()['label'].drop('label')
    numerical_corr_sorted = numerical_corr.abs().sort_values(ascending=False)
    
    # Take top 8 most correlated features
    top_features = numerical_corr_sorted.head(8)
    colors = ['red' if train_data[numerical_features + ['label']].corr()['label'][feat] > 0 else 'blue' 
              for feat in top_features.index]
    
    bars = plt.bar(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
    plt.title('Numerical Features Correlation with Label (Top 8)', fontsize=14, fontweight='bold')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient (Absolute)')
    plt.xticks(range(len(top_features)), top_features.index, rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Lead time vs cancellation rate
    plt.subplot(4, 3, 3)
    train_data['lead_time_group'] = pd.cut(train_data['lead_time'], 
                                          bins=[0, 7, 30, 90, 365], 
                                          labels=['0-7 days', '7-30 days', '30-90 days', '90+ days'])
    
    lead_time_analysis = train_data.groupby('lead_time_group')['label'].agg(['count', 'mean'])
    lead_time_analysis.columns = ['Booking Count', 'Cancellation Rate']
    
    bars = plt.bar(range(len(lead_time_analysis)), lead_time_analysis['Cancellation Rate'], 
                   color='salmon', alpha=0.7)
    plt.title('Lead Time vs Cancellation Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Lead Time')
    plt.ylabel('Cancellation Rate')
    plt.xticks(range(len(lead_time_analysis)), lead_time_analysis.index, rotation=45)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Room price vs cancellation rate
    plt.subplot(4, 3, 4)
    train_data['price_group'] = pd.cut(train_data['avg_price_per_room'], 
                                      bins=[0, 50, 100, 150, 200, 1000], 
                                      labels=['0-50', '50-100', '100-150', '150-200', '200+'])
    
    price_analysis = train_data.groupby('price_group')['label'].agg(['count', 'mean'])
    price_analysis.columns = ['Booking Count', 'Cancellation Rate']
    
    bars = plt.bar(range(len(price_analysis)), price_analysis['Cancellation Rate'], 
                   color='lightgreen', alpha=0.7)
    plt.title('Room Price vs Cancellation Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Price Range')
    plt.ylabel('Cancellation Rate')
    plt.xticks(range(len(price_analysis)), price_analysis.index, rotation=45)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Special requests vs cancellation rate
    plt.subplot(4, 3, 5)
    special_requests_analysis = train_data.groupby('no_of_special_requests')['label'].agg(['count', 'mean'])
    special_requests_analysis.columns = ['Booking Count', 'Cancellation Rate']
    
    bars = plt.bar(range(len(special_requests_analysis)), special_requests_analysis['Cancellation Rate'], 
                   color='purple', alpha=0.7)
    plt.title('Special Requests vs Cancellation Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Special Requests')
    plt.ylabel('Cancellation Rate')
    plt.xticks(range(len(special_requests_analysis)), special_requests_analysis.index)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 6. Repeated guest vs cancellation rate
    plt.subplot(4, 3, 6)
    repeated_guest_analysis = train_data.groupby('repeated_guest')['label'].agg(['count', 'mean'])
    repeated_guest_analysis.columns = ['Booking Count', 'Cancellation Rate']
    
    bars = plt.bar(range(len(repeated_guest_analysis)), repeated_guest_analysis['Cancellation Rate'], 
                   color='orange', alpha=0.7)
    plt.title('Repeated Guest vs Cancellation Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Repeated Guest (0=No, 1=Yes)')
    plt.ylabel('Cancellation Rate')
    plt.xticks(range(len(repeated_guest_analysis)), ['No', 'Yes'])
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 7. Market segment type vs cancellation rate
    plt.subplot(4, 3, 7)
    market_segment_analysis = train_data.groupby('market_segment_type')['label'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(market_segment_analysis)), market_segment_analysis.values, 
                   color='skyblue', alpha=0.7)
    plt.title('Market Segment Type vs Cancellation Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Market Segment Type')
    plt.ylabel('Cancellation Rate')
    plt.xticks(range(len(market_segment_analysis)), market_segment_analysis.index, rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 8. Room type vs cancellation rate
    plt.subplot(4, 3, 8)
    room_type_analysis = train_data.groupby('room_type_reserved')['label'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(room_type_analysis)), room_type_analysis.values, 
                   color='lightcoral', alpha=0.7)
    plt.title('Room Type vs Cancellation Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Room Type')
    plt.ylabel('Cancellation Rate')
    plt.xticks(range(len(room_type_analysis)), room_type_analysis.index, rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 9. Meal plan vs cancellation rate
    plt.subplot(4, 3, 9)
    meal_plan_analysis = train_data.groupby('type_of_meal_plan')['label'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(meal_plan_analysis)), meal_plan_analysis.values, 
                   color='lightgreen', alpha=0.7)
    plt.title('Meal Plan vs Cancellation Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Meal Plan')
    plt.ylabel('Cancellation Rate')
    plt.xticks(range(len(meal_plan_analysis)), meal_plan_analysis.index, rotation=45, ha='right')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 10. Numerical features correlation heatmap
    plt.subplot(4, 3, 10)
    correlation_matrix = train_data[numerical_features].corr()
    
    # Show only top 8 most correlated features
    top_numerical_features = numerical_corr_sorted.head(8).index.tolist()
    corr_subset = correlation_matrix.loc[top_numerical_features, top_numerical_features]
    
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Numerical Features Correlation Matrix (Top 8)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 11. Previous cancellations vs current cancellation rate
    plt.subplot(4, 3, 11)
    prev_cancellation_analysis = train_data.groupby('no_of_previous_cancellations')['label'].agg(['count', 'mean'])
    prev_cancellation_analysis.columns = ['Booking Count', 'Cancellation Rate']
    
    # Show only top 8 most common values
    top_prev_cancellations = prev_cancellation_analysis.head(8)
    
    bars = plt.bar(range(len(top_prev_cancellations)), top_prev_cancellations['Cancellation Rate'], 
                   color='red', alpha=0.7)
    plt.title('Previous Cancellations vs Current Cancellation Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Previous Cancellations')
    plt.ylabel('Current Cancellation Rate')
    plt.xticks(range(len(top_prev_cancellations)), top_prev_cancellations.index)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 12. Previous non-cancelled bookings vs current cancellation rate
    plt.subplot(4, 3, 12)
    prev_not_cancelled_analysis = train_data.groupby('no_of_previous_bookings_not_canceled')['label'].agg(['count', 'mean'])
    prev_not_cancelled_analysis.columns = ['Booking Count', 'Cancellation Rate']
    
    # Show only top 8 most common values
    top_prev_not_cancelled = prev_not_cancelled_analysis.head(8)
    
    bars = plt.bar(range(len(top_prev_not_cancelled)), top_prev_not_cancelled['Cancellation Rate'], 
                   color='green', alpha=0.7)
    plt.title('Previous Non-cancelled Bookings vs Current Cancellation Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Previous Non-cancelled Bookings')
    plt.ylabel('Current Cancellation Rate')
    plt.xticks(range(len(top_prev_not_cancelled)), top_prev_not_cancelled.index)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('hotel_booking_analysis_visualization_english.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization charts saved as 'hotel_booking_analysis_visualization_english.png'")

if __name__ == "__main__":
    create_visualizations()
