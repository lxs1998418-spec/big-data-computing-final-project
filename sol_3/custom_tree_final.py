#!/usr/bin/env python3
"""
Final Custom Decision Tree Implementation for Hotel Booking Cancellation Prediction
Optimized version with essential features and efficient implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
warnings.filterwarnings('ignore')

class TreeNode:
    """Node class for the decision tree"""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = value is not None

class CustomDecisionTree:
    """
    Custom Decision Tree implementation with entropy-based splitting
    """
    
    def __init__(self, max_depth=12, min_samples_split=30, min_samples_leaf=15, 
                 max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.feature_importance_ = None
        self.n_features_ = None
        
    def _entropy(self, y):
        """Calculate entropy"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        total = len(y)
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain"""
        parent_entropy = self._entropy(y)
        n_left, n_right = len(y_left), len(y_right)
        n_total = len(y)
        
        if n_total == 0:
            return 0
        
        left_entropy = self._entropy(y_left)
        right_entropy = self._entropy(y_right)
        
        weighted_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy
        return parent_entropy - weighted_entropy
    
    def _find_best_split(self, X, y, feature_indices):
        """Find best split for given features"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            values = X[:, feature_idx]
            unique_values = np.unique(values)
            
            for threshold in unique_values:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Build tree recursively"""
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return TreeNode(value=self._predict_leaf(y))
        
        # Select features
        n_features = X.shape[1]
        if self.max_features is None:
            feature_indices = list(range(n_features))
        elif self.max_features == 'sqrt':
            np.random.seed(self.random_state + depth)
            feature_indices = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        else:
            np.random.seed(self.random_state + depth)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, feature_indices)
        
        if best_feature is None or best_gain <= 0:
            return TreeNode(value=self._predict_leaf(y))
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(feature=best_feature, threshold=best_threshold, 
                       left=left_tree, right=right_tree)
    
    def _predict_leaf(self, y):
        """Predict class for leaf node"""
        counts = Counter(y)
        return counts.most_common(1)[0][0]
    
    def _predict_single(self, x, node):
        """Predict for single sample"""
        if node.is_leaf:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def fit(self, X, y):
        """Train the tree"""
        np.random.seed(self.random_state)
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y)
        self._calculate_feature_importance()
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.root))
        return np.array(predictions)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance"""
        self.feature_importance_ = np.zeros(self.n_features_)
        self._calculate_importance_recursive(self.root)
        
        # Normalize
        total = np.sum(self.feature_importance_)
        if total > 0:
            self.feature_importance_ /= total
    
    def _calculate_importance_recursive(self, node):
        """Recursively calculate feature importance"""
        if node.is_leaf:
            return
        
        if node.feature is not None:
            self.feature_importance_[node.feature] += 1
        
        self._calculate_importance_recursive(node.left)
        self._calculate_importance_recursive(node.right)

class CustomRandomForest:
    """
    Custom Random Forest using our decision trees
    """
    
    def __init__(self, n_estimators=50, max_depth=10, min_samples_split=20, 
                 min_samples_leaf=10, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_importance_ = None
        
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample"""
        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        """Train random forest"""
        np.random.seed(self.random_state)
        self.trees = []
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)
            
            # Train tree
            tree = CustomDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state + i
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        return self
    
    def predict(self, X):
        """Make predictions using majority voting"""
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        
        # Majority voting
        predictions = np.array(predictions)
        final_predictions = []
        for i in range(len(X)):
            votes = predictions[:, i]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        
        return np.array(final_predictions)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance across all trees"""
        if not self.trees:
            return
        
        importance = np.zeros(self.trees[0].n_features_)
        for tree in self.trees:
            importance += tree.feature_importance_
        
        self.feature_importance_ = importance / len(self.trees)

class HotelBookingPredictor:
    """
    Main predictor class with feature engineering based on analysis insights
    """
    
    def __init__(self, use_ensemble=True):
        self.use_ensemble = use_ensemble
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        
    def _load_data(self):
        """Load datasets"""
        print("Loading datasets...")
        self.train_df = pd.read_csv('train.csv')
        self.test_df = pd.read_csv('test.csv')
        self.sample_submission = pd.read_csv('sample_submission.csv')
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        
    def _feature_engineering(self, df, is_training=True):
        """
        Feature engineering based on data analysis insights
        """
        df = df.copy()
        
        # 1. Lead time features (most important from analysis)
        df['lead_time_log'] = np.log1p(df['lead_time'])
        df['lead_time_category'] = pd.cut(df['lead_time'], 
                                        bins=[0, 7, 30, 90, 365, float('inf')], 
                                        labels=['immediate', 'short', 'medium', 'long', 'very_long'])
        
        # 2. Price features
        df['price_log'] = np.log1p(df['avg_price_per_room'])
        df['price_category'] = pd.cut(df['avg_price_per_room'], 
                                     bins=[0, 50, 80, 120, 200, float('inf')], 
                                     labels=['budget', 'economy', 'standard', 'premium', 'luxury'])
        
        # 3. Temporal features
        df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        df['weekend_ratio'] = df['no_of_weekend_nights'] / (df['total_nights'] + 1)
        df['is_weekend_booking'] = (df['no_of_weekend_nights'] > 0).astype(int)
        
        # 4. Guest features
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        df['has_children'] = (df['no_of_children'] > 0).astype(int)
        
        # 5. Historical behavior (key insights from analysis)
        df['total_previous_bookings'] = (df['no_of_previous_cancellations'] + 
                                       df['no_of_previous_bookings_not_canceled'])
        df['cancellation_rate'] = np.where(
            df['total_previous_bookings'] > 0,
            df['no_of_previous_cancellations'] / df['total_previous_bookings'],
            0
        )
        df['is_repeated_guest'] = df['repeated_guest']
        df['has_cancellation_history'] = (df['no_of_previous_cancellations'] > 0).astype(int)
        
        # 6. Time features
        df['arrival_season'] = df['arrival_month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # 7. Interaction features (based on analysis insights)
        df['lead_time_price_interaction'] = df['lead_time'] * df['avg_price_per_room']
        df['guests_nights_interaction'] = df['total_guests'] * df['total_nights']
        
        # 8. Special request features
        df['has_special_requests'] = (df['no_of_special_requests'] > 0).astype(int)
        
        # 9. Market segment features
        df['is_online_booking'] = (df['market_segment_type'] == 'Online').astype(int)
        df['is_corporate_booking'] = (df['market_segment_type'] == 'Corporate').astype(int)
        
        # 10. Room type features
        df['is_room_type_1'] = (df['room_type_reserved'] == 'Room_Type 1').astype(int)
        df['is_room_type_4'] = (df['room_type_reserved'] == 'Room_Type 4').astype(int)
        df['is_room_type_6'] = (df['room_type_reserved'] == 'Room_Type 6').astype(int)
        
        # 11. Meal plan features
        df['has_meal_plan'] = (df['type_of_meal_plan'] != 'Not Selected').astype(int)
        
        return df
    
    def _encode_categorical_features(self, df, is_training=True):
        """Encode categorical features"""
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_columns:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    df[col] = df[col].astype(str)
                    known_values = self.label_encoders[col].classes_
                    
                    # Handle unseen categories
                    unseen_mask = ~df[col].isin(known_values)
                    if unseen_mask.any():
                        most_frequent = self.label_encoders[col].classes_[0]
                        df.loc[unseen_mask, col] = most_frequent
                    
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _prepare_features(self, df, is_training=True):
        """Prepare features"""
        # Feature engineering
        df = self._feature_engineering(df, is_training)
        
        # Encode categorical features
        df = self._encode_categorical_features(df, is_training)
        
        # Select features
        if is_training:
            feature_columns = [col for col in df.columns if col not in ['id', 'label']]
            self.feature_names = feature_columns
        else:
            feature_columns = self.feature_names
        
        X = df[feature_columns].values
        
        if is_training:
            y = df['label'].values
            return X, y
        else:
            return X
    
    def train(self):
        """Train the model"""
        print("Starting model training...")
        
        # Load data
        self._load_data()
        
        # Prepare features
        X_train, y_train = self._prepare_features(self.train_df, is_training=True)
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        # Choose model
        if self.use_ensemble:
            print("Training Custom Random Forest...")
            self.model = CustomRandomForest(
                n_estimators=50,
                max_depth=12,
                min_samples_split=30,
                min_samples_leaf=15,
                max_features='sqrt',
                random_state=42
            )
        else:
            print("Training Custom Decision Tree...")
            self.model = CustomDecisionTree(
                max_depth=15,
                min_samples_split=30,
                min_samples_leaf=15,
                max_features='sqrt',
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        # Training performance
        train_predictions = self.model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return self
    
    def evaluate_model(self):
        """Evaluate model with cross-validation"""
        print("\n=== Model Evaluation ===")
        
        X_train, y_train = self._prepare_features(self.train_df, is_training=True)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Wrapper for sklearn compatibility
        class SklearnWrapper:
            def __init__(self, model):
                self.model = model
            
            def fit(self, X, y):
                self.model.fit(X, y)
                return self
            
            def predict(self, X):
                return self.model.predict(X)
        
        wrapper = SklearnWrapper(self.model)
        
        # F1 scores
        f1_scores = cross_val_score(wrapper, X_train, y_train, cv=cv, scoring='f1_macro')
        
        print(f"Cross-validation F1 scores: {f1_scores}")
        print(f"Mean F1 score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
        
        # Feature importance
        if hasattr(self.model, 'feature_importance_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importance_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(importance_df.head(10))
        
        return f1_scores
    
    def predict(self):
        """Generate predictions"""
        print("\n=== Generating Predictions ===")
        
        X_test = self._prepare_features(self.test_df, is_training=False)
        predictions = self.model.predict(X_test)
        
        # Create submission
        submission = pd.DataFrame({
            'id': self.test_df['id'],
            'label': predictions
        })
        
        submission.to_csv('submission.csv', index=False)
        print("Predictions saved to submission.csv")
        
        # Prediction distribution
        pred_counts = Counter(predictions)
        print(f"Prediction distribution: {dict(pred_counts)}")
        
        return submission

def main():
    """Main function"""
    print("=== Custom Decision Tree for Hotel Booking Cancellation Prediction ===\n")
    
    # Train with ensemble
    print("Training with Custom Random Forest...")
    predictor = HotelBookingPredictor(use_ensemble=True)
    predictor.train()
    
    # Evaluate
    f1_scores = predictor.evaluate_model()
    
    # Generate predictions
    submission = predictor.predict()
    
    print(f"\n=== Final Results ===")
    print(f"Model: Custom Random Forest")
    print(f"Features: {len(predictor.feature_names)}")
    print(f"Mean F1 Score: {f1_scores.mean():.4f}")
    print(f"Predictions saved to: submission.csv")
    
    return predictor, submission

if __name__ == "__main__":
    predictor, submission = main()
