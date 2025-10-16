# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')


# 数据预处理函数
def preprocess_data(train_df, test_df):
    """优化的数据预处理函数"""
    train_processed = train_df.copy()
    test_processed = test_df.copy()

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

    # 特征工程 - 创建有意义的特征
    # 1. 可负担性比率
    train_processed['affordability_ratio'] = train_processed['customer_salary'] / (train_processed['price'] + 1)
    test_processed['affordability_ratio'] = test_processed['customer_salary'] / (test_processed['price'] + 1)

    # 2. 贷款价值比
    train_processed['loan_to_value'] = train_processed['loan_amount'] / (train_processed['price'] + 1)
    test_processed['loan_to_value'] = test_processed['loan_amount'] / (test_processed['price'] + 1)

    # 3. 房产年龄
    current_year = 2024
    train_processed['property_age'] = current_year - train_processed['constructed_year']
    test_processed['property_age'] = current_year - test_processed['constructed_year']

    # 4. 支付能力
    train_processed['payment_capacity'] = train_processed['customer_salary'] - train_processed['monthly_expenses']
    test_processed['payment_capacity'] = test_processed['customer_salary'] - test_processed['monthly_expenses']

    # 5. 首付比率
    train_processed['down_payment_ratio'] = train_processed['down_payment'] / (train_processed['price'] + 1)
    test_processed['down_payment_ratio'] = test_processed['down_payment'] / (test_processed['price'] + 1)

    # 6. 风险评分
    train_processed['risk_score'] = train_processed['crime_cases_reported'] + train_processed['legal_cases_on_property']
    test_processed['risk_score'] = test_processed['crime_cases_reported'] + test_processed['legal_cases_on_property']

    # 7. 质量评分
    train_processed['quality_score'] = train_processed['satisfaction_score'] + train_processed['neighbourhood_rating'] + \
                                       train_processed['connectivity_score']
    test_processed['quality_score'] = test_processed['satisfaction_score'] + test_processed['neighbourhood_rating'] + \
                                      test_processed['connectivity_score']

    return train_processed, test_processed, label_encoders


# 优化版决策树实现
class OptimizedDecisionTree:
    """
    优化版决策树实现 - 解决训练速度慢的问题
    """

    def __init__(self, max_depth=10, min_samples_split=50, min_samples_leaf=25,
                 max_features=None, criterion='entropy', random_state=42):
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
        counts = np.bincount(y)
        probabilities = counts / len(y)
        # 避免log(0)的问题
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def _gini_vectorized(self, y):
        """向量化基尼不纯度计算"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
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
        counts = np.bincount(y)
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
        if feature is None or gain <= 0:
            return self._create_leaf(y)

        # 分割数据
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

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

        print(f"开始训练决策树...")
        start_time = time.time()

        self.tree = self._build_tree_optimized(X, y)

        end_time = time.time()
        print(f"决策树训练完成，耗时: {end_time - start_time:.2f}秒")

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




if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42)

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")

    # 应用预处理
    print("预处理数据...")
    train_processed, test_processed, encoders = preprocess_data(train_df, test_df)

    # 准备特征和目标
    feature_cols = [col for col in train_processed.columns if col not in ['id', 'label']]
    X_train = train_processed[feature_cols]
    y_train = train_processed['label']
    X_test = test_processed[feature_cols]

    config = {
        'max_depth': 15,
        'min_samples_split': 50,
        'min_samples_leaf': 25,
        'max_features': None
    }

    # 分割数据用于验证
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"训练集大小: {X_train_split.shape[0]}")
    print(f"验证集大小: {X_val_split.shape[0]}")

    # 训练最终模型
    print("开始训练最终模型...")
    start_time = time.time()
    final_model = OptimizedDecisionTree(**config)
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

    submission_filename = 'submission.csv'
    submission_df.to_csv(submission_filename, index=False)

    print(f"提交文件 '{submission_filename}' 创建成功!")
    print(f"测试预测分布:")
    unique, counts = np.unique(test_predictions, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  类别 {label}: {count} 样本 ({count / len(test_predictions) * 100:.1f}%)")
