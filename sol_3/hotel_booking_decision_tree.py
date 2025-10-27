# 酒店预订取消预测 - 自定义决策树实现
# Hotel Booking Cancellation Prediction - Custom Decision Tree Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import time
import warnings
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 酒店预订数据分区间策略配置
HOTEL_BINNING_STRATEGIES = {
    # 高唯一值比例特征 - 必须分区间
    'avg_price_per_room': {
        'strategy': 'quantile',
        'n_bins': 8,
        'bin_edges': None  # 将在运行时计算
    },
    'lead_time': {
        'strategy': 'quantile', 
        'n_bins': 8,
        'bin_edges': None
    },
    'no_of_week_nights': {
        'strategy': 'equal_width',
        'n_bins': 6, 
        'bin_edges': None
    },
    'no_of_weekend_nights': {
        'strategy': 'equal_width',
        'n_bins': 6,
        'bin_edges': None
    },
    # 中等风险特征 - 适度分区间
    'no_of_adults': {
        'strategy': 'equal_width',
        'n_bins': 5,
        'bin_edges': None
    },
    'no_of_children': {
        'strategy': 'equal_width',
        'n_bins': 4,
        'bin_edges': None
    },
    'arrival_month': {
        'strategy': 'equal_width',
        'n_bins': 6,
        'bin_edges': None
    }
}

def apply_hotel_binning_strategies(df, binning_strategies, is_training=True):
    """
    应用分区间策略到酒店预订数据框
    
    Args:
        df: 输入数据框
        binning_strategies: 分区间策略配置
        is_training: 是否为训练数据（用于计算分区间边界）
    
    Returns:
        处理后的数据框和分区间边界
    """
    df_binned = df.copy()
    bin_edges_dict = {}
    
    for feature, strategy_info in binning_strategies.items():
        if feature not in df.columns:
            continue
            
        data = df[feature]
        strategy = strategy_info['strategy']
        n_bins = strategy_info['n_bins']
        
        if strategy == 'equal_width':
            # 等宽分区间
            bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
        elif strategy == 'quantile':
            # 分位数分区间
            bin_edges = [data.quantile(i/n_bins) for i in range(n_bins + 1)]
            bin_edges[0] = data.min()  # 确保包含最小值
            bin_edges[-1] = data.max()  # 确保包含最大值
        else:
            continue
            
        # 应用分区间
        try:
            binned_data = pd.cut(data, bins=bin_edges, include_lowest=True, duplicates='drop')
            # 转换为数值标签
            df_binned[feature] = binned_data.cat.codes
            bin_edges_dict[feature] = bin_edges
            
            if is_training:
                print(f"  {feature}: {strategy}分区间, {n_bins}个区间, 边界: {[f'{x:.2f}' for x in bin_edges]}")
                
        except Exception as e:
            print(f"  {feature}: 分区间失败 - {e}")
            continue
    
    return df_binned, bin_edges_dict

# 酒店预订数据预处理函数
def preprocess_hotel_data(train_df, test_df):
    """酒店预订数据预处理函数 - 包含分区间处理"""
    train_processed = train_df.copy()
    test_processed = test_df.copy()

    print("开始酒店预订数据预处理...")
    
    # 1. 处理分类变量
    print("1. 处理分类变量...")
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        combined_data = pd.concat([train_processed[col], test_processed[col]])
        le.fit(combined_data)
        train_processed[col] = le.transform(train_processed[col])
        test_processed[col] = le.transform(test_processed[col])
        label_encoders[col] = le

    # 2. 应用分区间策略
    print("2. 应用分区间策略...")
    train_processed, bin_edges_dict = apply_hotel_binning_strategies(
        train_processed, HOTEL_BINNING_STRATEGIES, is_training=True
    )
    
    # 为测试数据使用相同的分区间边界
    for feature, bin_edges in bin_edges_dict.items():
        if feature in test_processed.columns:
            try:
                binned_data = pd.cut(test_processed[feature], bins=bin_edges, 
                                   include_lowest=True, duplicates='drop')
                test_processed[feature] = binned_data.cat.codes
            except Exception as e:
                print(f"  测试数据{feature}分区间失败 - {e}")
                continue

    # 3. 特征工程 - 创建有意义的特征
    print("3. 创建衍生特征...")
    
    # 1. 总住宿天数
    train_processed['total_nights'] = train_processed['no_of_week_nights'] + train_processed['no_of_weekend_nights']
    test_processed['total_nights'] = test_processed['no_of_week_nights'] + test_processed['no_of_weekend_nights']

    # 2. 总客人数量
    train_processed['total_guests'] = train_processed['no_of_adults'] + train_processed['no_of_children']
    test_processed['total_guests'] = test_processed['no_of_adults'] + test_processed['no_of_children']

    # 3. 预订提前期分类
    def categorize_lead_time(lead_time):
        if lead_time <= 7:
            return 0  # 短期预订
        elif lead_time <= 30:
            return 1  # 中期预订
        elif lead_time <= 90:
            return 2  # 长期预订
        else:
            return 3  # 超长期预订
    
    train_processed['lead_time_category'] = train_processed['lead_time'].apply(categorize_lead_time)
    test_processed['lead_time_category'] = test_processed['lead_time'].apply(categorize_lead_time)

    # 4. 价格区间分类
    def categorize_price(price):
        if price <= 50:
            return 0  # 低价
        elif price <= 100:
            return 1  # 中低价
        elif price <= 150:
            return 2  # 中价
        elif price <= 200:
            return 3  # 中高价
        else:
            return 4  # 高价
    
    train_processed['price_category'] = train_processed['avg_price_per_room'].apply(categorize_price)
    test_processed['price_category'] = test_processed['avg_price_per_room'].apply(categorize_price)

    # 5. 客户忠诚度评分
    train_processed['loyalty_score'] = train_processed['repeated_guest'] * 2 + \
                                      train_processed['no_of_previous_bookings_not_canceled'] - \
                                      train_processed['no_of_previous_cancellations']
    test_processed['loyalty_score'] = test_processed['repeated_guest'] * 2 + \
                                     test_processed['no_of_previous_bookings_not_canceled'] - \
                                     test_processed['no_of_previous_cancellations']

    # 6. 风险评分
    train_processed['risk_score'] = train_processed['lead_time_category'] + \
                                   train_processed['price_category'] - \
                                   train_processed['no_of_special_requests']
    test_processed['risk_score'] = test_processed['lead_time_category'] + \
                                  test_processed['price_category'] - \
                                  test_processed['no_of_special_requests']

    # 7. 季节性特征
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # 冬季
        elif month in [3, 4, 5]:
            return 1  # 春季
        elif month in [6, 7, 8]:
            return 2  # 夏季
        else:
            return 3  # 秋季
    
    train_processed['season'] = train_processed['arrival_month'].apply(get_season)
    test_processed['season'] = test_processed['arrival_month'].apply(get_season)

    # 4. 对衍生特征应用分区间策略
    print("4. 对衍生特征应用分区间策略...")
    
    # 衍生特征的分区间策略
    derived_binning_strategies = {
        'total_nights': {'strategy': 'equal_width', 'n_bins': 6},
        'total_guests': {'strategy': 'equal_width', 'n_bins': 5},
        'loyalty_score': {'strategy': 'quantile', 'n_bins': 6},
        'risk_score': {'strategy': 'quantile', 'n_bins': 6}
    }
    
    # 对训练数据应用衍生特征分区间
    train_processed, derived_bin_edges = apply_hotel_binning_strategies(
        train_processed, derived_binning_strategies, is_training=True
    )
    
    # 对测试数据应用相同的分区间边界
    for feature, bin_edges in derived_bin_edges.items():
        if feature in test_processed.columns:
            try:
                binned_data = pd.cut(test_processed[feature], bins=bin_edges, 
                                   include_lowest=True, duplicates='drop')
                test_processed[feature] = binned_data.cat.codes
            except Exception as e:
                print(f"  测试数据{feature}分区间失败 - {e}")
                continue

    print("酒店预订数据预处理完成!")
    return train_processed, test_processed, label_encoders


# 酒店预订优化版决策树实现
class HotelBookingDecisionTree:
    """
    酒店预订取消预测优化版决策树实现
    """

    def __init__(self, max_depth=12, min_samples_split=30, min_samples_leaf=15,
                 max_features='sqrt', criterion='entropy', random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.tree = None
        self.feature_names = None

    def _entropy_vectorized(self, y):
        """向量化熵计算 - 比循环快很多"""
        if len(y) == 0:
            return 0
        # 使用bincount和向量化操作
        counts = np.bincount(y, minlength=2)  # 确保至少有两个类别
        probabilities = counts / len(y)
        # 避免log(0)的问题
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def _gini_vectorized(self, y):
        """向量化基尼不纯度计算"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y, minlength=2)  # 确保至少有两个类别
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _get_feature_subset(self, n_features):
        """特征采样 - 减少每次分割考虑的特征数量"""
        if self.max_features is None:
            return np.arange(n_features)
        elif self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_selected = int(np.log2(n_features))
        else:
            n_selected = min(self.max_features, n_features)

        np.random.seed(self.random_state)
        return np.random.choice(n_features, size=n_selected, replace=False)

    def _find_best_split_optimized(self, X, y):
        """优化的最佳分割查找 - 使用向量化操作"""
        best_gain = 0
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        n_samples = X.shape[0]

        # 特征采样 - 关键优化点
        feature_subset = self._get_feature_subset(n_features)

        for feature in feature_subset:
            # 获取该特征的所有唯一值作为候选阈值
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)

            # 如果唯一值太少，跳过
            if len(unique_values) < 2:
                continue

            # 向量化计算每个阈值的信息增益
            for threshold in unique_values[1:]:  # 跳过第一个值
                # 向量化分割
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # 检查最小样本数条件
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                # 计算信息增益
                if self.criterion == 'entropy':
                    parent_impurity = self._entropy_vectorized(y)
                    left_impurity = self._entropy_vectorized(y_left)
                    right_impurity = self._entropy_vectorized(y_right)
                else:  # gini
                    parent_impurity = self._gini_vectorized(y)
                    left_impurity = self._gini_vectorized(y_left)
                    right_impurity = self._gini_vectorized(y_right)

                # 加权平均
                n_left, n_right = len(y_left), len(y_right)
                weighted_impurity = (n_left / n_samples) * left_impurity + (n_right / n_samples) * right_impurity
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _create_leaf(self, y):
        """创建叶节点"""
        counts = np.bincount(y, minlength=2)  # 确保至少有两个类别
        return np.argmax(counts)

    def _build_tree_optimized(self, X, y, depth=0):
        """优化的树构建 - 减少递归深度和计算量"""
        # 早期停止条件 - 关键优化点
        if (depth >= self.max_depth or
                len(y) < self.min_samples_split or
                len(np.unique(y)) == 1):
            return self._create_leaf(y)

        # 查找最佳分割
        feature, threshold, gain = self._find_best_split_optimized(X, y)

        # 如果没有好的分割，创建叶节点
        if feature is None or gain <= 0.001:  # 降低增益阈值
            return self._create_leaf(y)

        # 分割数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # 检查分割后的样本数
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self._create_leaf(y)

        # 递归构建子树
        left_subtree = self._build_tree_optimized(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree_optimized(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        """训练决策树"""
        X = np.array(X)
        y = np.array(y)

        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        print(f"开始训练酒店预订决策树...")
        start_time = time.time()

        self.tree = self._build_tree_optimized(X, y)

        end_time = time.time()
        print(f"酒店预订决策树训练完成，耗时: {end_time - start_time:.2f}秒")

        return self

    def _predict_sample(self, x, tree):
        """预测单个样本"""
        if isinstance(tree, dict):
            if x[tree['feature']] <= tree['threshold']:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])
        else:
            return tree

    def predict(self, X):
        """批量预测"""
        X = np.array(X)
        predictions = []
        for x in X:
            predictions.append(self._predict_sample(x, self.tree))
        return np.array(predictions)

    def predict_proba(self, X):
        """预测概率"""
        predictions = self.predict(X)
        probas = np.zeros((len(X), 2))
        probas[np.arange(len(X)), predictions] = 1
        return probas

    def get_feature_importance(self):
        """计算特征重要性"""
        if self.tree is None:
            return None
        
        feature_importance = np.zeros(len(self.feature_names) if self.feature_names else 0)
        self._calculate_importance(self.tree, feature_importance)
        
        # 归一化
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        return feature_importance

    def _calculate_importance(self, node, feature_importance):
        """递归计算特征重要性"""
        if isinstance(node, dict):
            feature_idx = node['feature']
            if feature_idx < len(feature_importance):
                feature_importance[feature_idx] += 1
            
            self._calculate_importance(node['left'], feature_importance)
            self._calculate_importance(node['right'], feature_importance)

    def visualize_tree(self, max_depth=3, figsize=(20, 12)):
        """可视化决策树结构"""
        if self.tree is None:
            print("模型尚未训练，请先调用fit方法")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 计算树的深度和节点位置
        tree_info = self._get_tree_info(self.tree, max_depth)
        
        # 绘制树结构
        self._draw_tree(ax, self.tree, 0.5, 0.9, 0.4, 0.1, max_depth, tree_info)
        
        plt.title('酒店预订取消预测决策树结构可视化', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def _get_tree_info(self, node, max_depth, depth=0):
        """获取树的基本信息"""
        if depth >= max_depth or not isinstance(node, dict):
            return {'depth': depth, 'is_leaf': True}
        
        left_info = self._get_tree_info(node['left'], max_depth, depth + 1)
        right_info = self._get_tree_info(node['right'], max_depth, depth + 1)
        
        return {
            'depth': depth,
            'is_leaf': False,
            'left': left_info,
            'right': right_info
        }

    def _draw_tree(self, ax, node, x, y, width, height, max_depth, tree_info, depth=0):
        """递归绘制树节点"""
        if depth >= max_depth:
            return
        
        # 节点颜色
        if isinstance(node, dict):
            # 内部节点 - 蓝色
            color = '#4A90E2'
            text_color = 'white'
            
            # 获取特征名称
            feature_name = self.feature_names[node['feature']] if self.feature_names else f'Feature_{node["feature"]}'
            node_text = f'{feature_name}\n≤ {node["threshold"]:.2f}'
        else:
            # 叶节点 - 绿色表示不取消，红色表示取消
            color = '#7ED321' if node == 0 else '#F5A623'
            text_color = 'white'
            node_text = f'预测: {"不取消" if node == 0 else "取消"}'
        
        # 绘制节点
        bbox = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(bbox)
        
        # 添加文本
        ax.text(x, y, node_text, ha='center', va='center', 
                fontsize=10, fontweight='bold', color=text_color)
        
        # 如果不是叶节点且未达到最大深度，绘制子节点
        if isinstance(node, dict) and depth < max_depth - 1:
            # 计算子节点位置
            child_width = width * 0.6
            child_height = height * 0.8
            child_y = y - height * 1.5
            
            # 左子节点
            left_x = x - width * 0.3
            right_x = x + width * 0.3
            
            # 绘制连接线
            ax.plot([x, left_x], [y - height/2, child_y + child_height/2], 
                   'k-', linewidth=2)
            ax.plot([x, right_x], [y - height/2, child_y + child_height/2], 
                   'k-', linewidth=2)
            
            # 添加标签
            ax.text((x + left_x)/2, (y + child_y)/2, '是', ha='center', va='center',
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.1", facecolor='lightblue'))
            ax.text((x + right_x)/2, (y + child_y)/2, '否', ha='center', va='center',
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.1", facecolor='lightcoral'))
            
            # 递归绘制子节点
            self._draw_tree(ax, node['left'], left_x, child_y, child_width, child_height, 
                          max_depth, tree_info, depth + 1)
            self._draw_tree(ax, node['right'], right_x, child_y, child_width, child_height, 
                          max_depth, tree_info, depth + 1)

    def plot_feature_importance(self, top_n=15, figsize=(12, 8)):
        """绘制特征重要性图"""
        importance = self.get_feature_importance()
        if importance is None:
            print("无法计算特征重要性")
            return
        
        # 获取特征名称
        feature_names = self.feature_names if self.feature_names else [f'Feature_{i}' for i in range(len(importance))]
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # 取前top_n个特征
        top_features = importance_df.tail(top_n)
        
        # 绘制水平条形图
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('特征重要性', fontsize=12)
        plt.title(f'酒店预订决策树特征重要性 (Top {top_n})', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # 添加数值标签
        for i, (idx, row) in enumerate(top_features.iterrows()):
            plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return top_features

    def plot_confusion_matrix(self, y_true, y_pred, figsize=(8, 6)):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['不取消', '取消'], 
                   yticklabels=['不取消', '取消'])
        plt.title('酒店预订取消预测混淆矩阵', fontsize=14, fontweight='bold')
        plt.xlabel('预测值', fontsize=12)
        plt.ylabel('真实值', fontsize=12)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)

    # 读取酒店预订数据
    train_df = pd.read_csv('sol_3/train.csv')
    test_df = pd.read_csv('sol_3/test.csv')

    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    
    # 查看标签分布
    print(f"\n标签分布:")
    label_counts = train_df['label'].value_counts()
    print(f"不取消 (0): {label_counts[0]} ({label_counts[0]/len(train_df)*100:.2f}%)")
    print(f"取消 (1): {label_counts[1]} ({label_counts[1]/len(train_df)*100:.2f}%)")

    # 应用预处理
    print("\n预处理数据...")
    train_processed, test_processed, encoders = preprocess_hotel_data(train_df, test_df)

    # 准备特征和目标
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'label']]
    X_train = train_processed[feature_cols]
    y_train = train_processed['label']
    X_test = test_processed[feature_cols]

    print(f"\n特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols}")

    # 决策树配置
    config = {
        'max_depth': 18,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': None,  # 使用所有特征
        'criterion': 'entropy'
    }

    # 分割数据用于验证
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\n训练集大小: {X_train_split.shape[0]}")
    print(f"验证集大小: {X_val_split.shape[0]}")

    # 训练最终模型
    print("\n开始训练最终模型...")
    start_time = time.time()
    final_model = HotelBookingDecisionTree(**config)
    final_model.fit(X_train_split, y_train_split)
    training_time = time.time() - start_time

    print(f"模型训练完成，耗时: {training_time:.2f}秒")

    # 验证模型性能
    val_predictions = final_model.predict(X_val_split)
    val_accuracy = np.mean(val_predictions == y_val_split)
    val_macro_f1 = f1_score(y_val_split, val_predictions, average='macro')

    print(f"\n验证集性能:")
    print(f"准确率: {val_accuracy:.4f}")
    print(f"Macro-F1: {val_macro_f1:.4f}")

    # 绘制混淆矩阵
    print("\n绘制验证集混淆矩阵...")
    final_model.plot_confusion_matrix(y_val_split, val_predictions)

    print(f"\n生成测试集预测...")
    start_time = time.time()
    test_predictions = final_model.predict(X_test)
    prediction_time = time.time() - start_time

    print(f"测试集预测完成，耗时: {prediction_time:.2f}秒")

    # 创建提交文件
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': test_predictions
    })

    submission_filename = 'hotel_booking_submission.csv'
    submission_df.to_csv(submission_filename, index=False)

    print(f"提交文件 '{submission_filename}' 创建成功!")
    print(f"测试预测分布:")
    unique, counts = np.unique(test_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count} 样本 ({count / len(test_predictions) * 100:.1f}%)")
    
    # 可视化决策树
    print("\n" + "="*50)
    print("开始可视化决策树...")
    
    # 1. 可视化决策树结构 (前3层)
    print("1. 绘制决策树结构图...")
    final_model.visualize_tree(max_depth=3, figsize=(20, 12))
    
    # 2. 绘制特征重要性图
    print("2. 绘制特征重要性图...")
    top_features = final_model.plot_feature_importance(top_n=15, figsize=(12, 8))
    
    # 3. 打印特征重要性信息
    print("\n特征重要性排序 (Top 10):")
    print("-" * 40)
    for i, (idx, row) in enumerate(top_features.tail(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    print("\n酒店预订决策树可视化完成!")
