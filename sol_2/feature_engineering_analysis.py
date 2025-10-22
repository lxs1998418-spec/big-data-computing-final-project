#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程分析：类别型特征标签化 + 连续特征分箱 + 新特征创建 + 热力图分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("="*60)
    print("特征工程分析：类别型特征标签化 + 连续特征分箱 + 新特征创建")
    print("="*60)
    
    # 1. 读取数据
    df = pd.read_csv('train.csv')
    print(f"原始数据形状: {df.shape}")
    print("原始特征类型:")
    print(df.dtypes.value_counts())
    
    # 2. 类别型特征标签化
    print("\n" + "="*60)
    print("1. 类别型特征标签化")
    print("="*60)
    
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    print(f"类别型特征: {categorical_features}")
    
    label_encoders = {}
    df_encoded = df.copy()
    
    for feature in categorical_features:
        le = LabelEncoder()
        df_encoded[feature + '_encoded'] = le.fit_transform(df[feature])
        label_encoders[feature] = le
        print(f"{feature} -> {feature}_encoded")
        print(f"  原始类别: {df[feature].unique()}")
        print(f"  编码后: {df_encoded[feature + '_encoded'].unique()}")
        print()
    
    df_encoded = df_encoded.drop(columns=categorical_features)
    print(f"标签化后的数据形状: {df_encoded.shape}")
    
    # 3. 连续数值特征分箱处理
    print("\n" + "="*60)
    print("2. 连续数值特征分箱处理")
    print("="*60)
    
    continuous_features = [col for col in df_encoded.select_dtypes(include=[np.number]).columns 
                          if col not in ['id', 'label']]
    
    print(f"连续特征数量: {len(continuous_features)}")
    
    def create_bins(df, feature, n_bins=5, method='quantile'):
        if method == 'quantile':
            df[feature + '_binned'] = pd.qcut(df[feature], q=n_bins, duplicates='drop', labels=False)
        else:
            df[feature + '_binned'] = pd.cut(df[feature], bins=n_bins, labels=False, include_lowest=True)
        return df
    
    df_binned = df_encoded.copy()
    binning_info = {}
    
    for feature in continuous_features:
        if feature in ['price', 'loan_amount', 'down_payment', 'customer_salary']:
            method = 'quantile'
            n_bins = 5
        elif feature in ['property_size_sqft', 'rooms', 'bathrooms']:
            method = 'uniform'
            n_bins = 4
        else:
            method = 'quantile'
            n_bins = 4
        
        df_binned = create_bins(df_binned, feature, n_bins, method)
        binning_info[feature] = {'method': method, 'n_bins': n_bins}
        print(f"{feature}: {method}分箱, {n_bins}个区间")
    
    print(f"分箱后的数据形状: {df_binned.shape}")
    
    # 4. 创建新的特征工程特征
    print("\n" + "="*60)
    print("3. 创建新的特征工程特征")
    print("="*60)
    
    df_features = df_binned.copy()
    
    # 4.1 房价相关比率特征
    print("3.1 房价相关比率特征")
    df_features['price_per_sqft'] = df_features['price'] / df_features['property_size_sqft']
    df_features['loan_to_price_ratio'] = df_features['loan_amount'] / df_features['price']
    df_features['down_payment_ratio'] = df_features['down_payment'] / df_features['price']
    df_features['salary_to_price_ratio'] = df_features['customer_salary'] / df_features['price']
    
    # 4.2 房屋特征组合
    print("3.2 房屋特征组合")
    df_features['total_rooms'] = df_features['rooms'] + df_features['bathrooms']
    df_features['room_density'] = df_features['total_rooms'] / df_features['property_size_sqft']
    df_features['has_garage_garden'] = (df_features['garage'] + df_features['garden']).astype(int)
    
    # 4.3 财务特征组合
    print("3.3 财务特征组合")
    df_features['monthly_income_ratio'] = df_features['customer_salary'] / 12 / df_features['monthly_expenses']
    df_features['loan_affordability'] = df_features['customer_salary'] / df_features['loan_amount']
    df_features['expense_ratio'] = df_features['monthly_expenses'] / df_features['customer_salary']
    
    # 4.4 风险特征组合
    print("3.4 风险特征组合")
    df_features['total_risk_cases'] = df_features['crime_cases_reported'] + df_features['legal_cases_on_property']
    df_features['risk_score'] = (df_features['total_risk_cases'] + 
                               (10 - df_features['neighbourhood_rating']) + 
                               (10 - df_features['connectivity_score'])) / 3
    
    # 4.5 时间特征
    print("3.5 时间特征")
    df_features['property_age'] = 2024 - df_features['constructed_year']
    df_features['is_new_property'] = (df_features['property_age'] <= 5).astype(int)
    df_features['is_old_property'] = (df_features['property_age'] >= 20).astype(int)
    
    print(f"特征工程后的数据形状: {df_features.shape}")
    
    # 5. 特征与特征之间的关系热力图分析
    print("\n" + "="*60)
    print("4. 特征与特征之间的关系热力图分析")
    print("="*60)
    
    numeric_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'id']
    
    print(f"参与相关性分析的特征数量: {len(numeric_features)}")
    
    correlation_matrix = df_features[numeric_features].corr()
    
    # 5.1 整体相关性热力图
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                cbar_kws={"shrink": .8})
    plt.title('特征与特征之间的相关性热力图', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5.2 高相关性特征识别
    print("\n高相关性特征识别 (|相关系数| > 0.7):")
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df = high_corr_df.sort_values('correlation', key=abs, ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        print("未发现高相关性特征对 (|相关系数| > 0.7)")
    
    # 5.3 与标签相关性最高的特征
    label_correlations = correlation_matrix['label'].drop('label').abs().sort_values(ascending=False)
    print(f"\n与标签相关性最高的前20个特征:")
    print(label_correlations.head(20).to_string())
    
    # 6. 特征与标签关系的详细分析
    print("\n" + "="*60)
    print("5. 特征与标签关系的详细分析")
    print("="*60)
    
    # 6.1 与标签相关性最高的特征可视化
    top_features = label_correlations.head(10).index.tolist()
    print(f"与标签相关性最高的前10个特征: {top_features}")
    
    # 创建子图显示这些特征与标签的关系
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_features):
        if i < 10:
            if df_features[feature].nunique() <= 10:
                feature_label_counts = df_features.groupby([feature, 'label']).size().unstack(fill_value=0)
                feature_label_counts.plot(kind='bar', ax=axes[i], stacked=True)
                axes[i].set_title(f'{feature} vs Label')
                axes[i].legend(['不买', '买'])
            else:
                df_features.boxplot(column=feature, by='label', ax=axes[i])
                axes[i].set_title(f'{feature} vs Label')
                axes[i].set_xlabel('Label')
            
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('top_features_vs_label.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6.2 新创建特征与标签的关系分析
    print("\n新创建特征与标签的关系分析:")
    new_features = ['price_per_sqft', 'loan_to_price_ratio', 'down_payment_ratio', 
                   'salary_to_price_ratio', 'total_rooms', 'room_density', 
                   'has_garage_garden', 'monthly_income_ratio', 'loan_affordability', 
                   'expense_ratio', 'total_risk_cases', 'risk_score', 'property_age']
    
    new_feature_correlations = {}
    for feature in new_features:
        if feature in df_features.columns:
            corr = df_features[feature].corr(df_features['label'])
            new_feature_correlations[feature] = corr
    
    new_corr_df = pd.DataFrame(list(new_feature_correlations.items()), 
                              columns=['Feature', 'Correlation_with_Label'])
    new_corr_df = new_corr_df.sort_values('Correlation_with_Label', key=abs, ascending=False)
    print(new_corr_df.to_string(index=False))
    
    # 7. 特征重要性分析
    print("\n" + "="*60)
    print("6. 特征重要性分析")
    print("="*60)
    
    # 准备数据
    X = df_features.drop(['id', 'label'], axis=1)
    y = df_features['label']
    
    # 处理无穷大和NaN值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("随机森林特征重要性 (前20个):")
    print(feature_importance.head(20).to_string(index=False))
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_20_features = feature_importance.head(20)
    plt.barh(range(len(top_20_features)), top_20_features['importance'])
    plt.yticks(range(len(top_20_features)), top_20_features['feature'])
    plt.xlabel('特征重要性')
    plt.title('随机森林特征重要性排序 (前20个)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算模型性能
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n随机森林模型准确率: {accuracy:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['不买', '买']))
    
    # 8. 特征工程总结和建议
    print("\n" + "="*60)
    print("7. 特征工程总结和建议")
    print("="*60)
    
    print("7.1 特征工程总结:")
    print(f"✓ 类别型特征标签化: {len(categorical_features)} 个特征")
    print(f"✓ 连续特征分箱处理: {len(continuous_features)} 个特征")
    print(f"✓ 新特征创建: {len(new_features)} 个特征")
    print(f"✓ 总特征数量: {X.shape[1]} 个")
    
    print("\n7.2 关键发现:")
    print("1. 与标签相关性最高的特征:")
    top_5_corr = label_correlations.head(5)
    for feature, corr in top_5_corr.items():
        print(f"   - {feature}: {corr:.4f}")
    
    print("\n2. 最重要的特征 (随机森林):")
    top_5_importance = feature_importance.head(5)
    for _, row in top_5_importance.iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")
    
    print("\n3. 新创建特征中最重要的:")
    new_feature_importance = feature_importance[feature_importance['feature'].isin(new_features)]
    if not new_feature_importance.empty:
        top_new_features = new_feature_importance.head(3)
        for _, row in top_new_features.iterrows():
            print(f"   - {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n7.3 模型性能评估:")
    print(f"随机森林准确率: {accuracy:.4f}")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    
    print("\n7.4 后续建议:")
    print("1. 特征选择:")
    print("   - 可以基于特征重要性进行特征选择")
    print("   - 移除重要性低于阈值的特征")
    print("   - 考虑特征之间的相关性，移除冗余特征")
    
    print("\n2. 模型优化:")
    print("   - 尝试不同的机器学习算法")
    print("   - 进行超参数调优")
    print("   - 使用交叉验证评估模型稳定性")
    
    print("\n3. 特征工程优化:")
    print("   - 可以尝试更多的特征组合")
    print("   - 考虑时间序列特征")
    print("   - 探索非线性特征变换")
    
    # 保存处理后的数据
    print(f"\n7.5 数据保存:")
    df_features.to_csv('train_features_engineered.csv', index=False)
    print("特征工程后的数据已保存为 'train_features_engineered.csv'")
    
    print("\n特征工程分析完成！")

if __name__ == "__main__":
    main()
