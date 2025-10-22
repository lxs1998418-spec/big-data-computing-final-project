#!/usr/bin/env python3
"""
Quick analysis summary for hotel booking cancellation prediction dataset
"""

import pandas as pd
import numpy as np

def main():
    # Load datasets
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print("=== HOTEL BOOKING CANCELLATION PREDICTION - DATASET ANALYSIS ===\n")
    
    # Basic dataset info
    print("📊 DATASET OVERVIEW:")
    print(f"   • Training samples: {len(train_df):,}")
    print(f"   • Test samples: {len(test_df):,}")
    print(f"   • Total features: {len(train_df.columns) - 2}")
    print(f"   • Feature types: {train_df.dtypes.value_counts().to_dict()}")
    
    # Target variable analysis
    label_counts = train_df['label'].value_counts()
    print(f"\n🎯 TARGET VARIABLE ANALYSIS:")
    print(f"   • Not Canceled (0): {label_counts[0]:,} ({label_counts[0]/len(train_df)*100:.1f}%)")
    print(f"   • Canceled (1): {label_counts[1]:,} ({label_counts[1]/len(train_df)*100:.1f}%)")
    print(f"   • Class imbalance ratio: {label_counts[1] / label_counts[0]:.3f}")
    
    # Data quality
    print(f"\n🔍 DATA QUALITY:")
    print(f"   • Missing values in training: {train_df.isnull().sum().sum()}")
    print(f"   • Missing values in test: {test_df.isnull().sum().sum()}")
    print(f"   • Duplicate rows in training: {train_df.duplicated().sum()}")
    print(f"   • Duplicate rows in test: {test_df.duplicated().sum()}")
    
    # Feature analysis
    numerical_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['id', 'label']]
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n📈 FEATURE ANALYSIS:")
    print(f"   • Numerical features: {len(numerical_cols)}")
    print(f"   • Categorical features: {len(categorical_cols)}")
    print(f"   • Numerical: {numerical_cols}")
    print(f"   • Categorical: {categorical_cols}")
    
    # Key correlations
    correlation_matrix = train_df[numerical_cols + ['label']].corr()
    correlations_with_target = correlation_matrix['label'].drop('label').sort_values(key=abs, ascending=False)
    
    print(f"\n🔗 TOP CORRELATIONS WITH TARGET:")
    for i, (feature, corr) in enumerate(correlations_with_target.head(5).items()):
        print(f"   {i+1}. {feature}: {corr:.3f}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • This is a binary classification problem")
    print(f"   • Moderate class imbalance (not severe)")
    print(f"   • Clean dataset with no missing values")
    print(f"   • Lead time and price are likely important predictors")
    print(f"   • Evaluation metric: Macro-F1 score")
    print(f"   • Ready for feature engineering and model development")
    
    # Cancellation rates by key features
    print(f"\n📊 CANCELLATION RATES BY KEY FEATURES:")
    
    # By market segment
    if 'market_segment_type' in train_df.columns:
        market_cancellation = train_df.groupby('market_segment_type')['label'].mean().sort_values(ascending=False)
        print(f"   • By Market Segment:")
        for segment, rate in market_cancellation.items():
            print(f"     - {segment}: {rate:.3f}")
    
    # By room type
    if 'room_type_reserved' in train_df.columns:
        room_cancellation = train_df.groupby('room_type_reserved')['label'].mean().sort_values(ascending=False)
        print(f"   • By Room Type:")
        for room, rate in room_cancellation.items():
            print(f"     - {room}: {rate:.3f}")
    
    # By lead time ranges
    if 'lead_time' in train_df.columns:
        train_df['lead_time_category'] = pd.cut(train_df['lead_time'], 
                                               bins=[0, 30, 90, 365, float('inf')], 
                                               labels=['0-30 days', '31-90 days', '91-365 days', '365+ days'])
        lead_cancellation = train_df.groupby('lead_time_category')['label'].mean()
        print(f"   • By Lead Time:")
        for category, rate in lead_cancellation.items():
            print(f"     - {category}: {rate:.3f}")
    
    print(f"\n✅ Analysis completed! Ready for model development.")

if __name__ == "__main__":
    main()
