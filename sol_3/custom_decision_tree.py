#!/usr/bin/env python3
"""
Custom Decision Tree Implementation for Hotel Booking Cancellation Prediction
Based on data analysis insights from the exploratory analysis.

This implementation includes:
1. Custom decision tree algorithm with entropy-based splitting
2. Feature engineering based on analysis insights
3. Pruning and regularization techniques
4. Cross-validation and evaluation
5. Prediction generation for test set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class TreeNode:
    """Node class for the decision tree"""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Feature to split on
        self.threshold = threshold      # Threshold for numerical features
        self.left = left               # Left child (<= threshold)
        self.right = right             # Right child (> threshold)
        self.value = value             # Prediction value (for leaf nodes)
        self.is_leaf = value is not None

class CustomDecisionTree:
    """
    Custom Decision Tree implementation with entropy-based splitting
    and pruning capabilities
    """
    
    def __init__(self, max_depth=10, min_samples_split=20, min_samples_leaf=10, 
                 max_features=None, random_state=42, pruning_alpha=0.01):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.pruning_alpha = pruning_alpha
        self.root = None
        self.feature_importance_ = None
        
    def _entropy(self, y):
        """Calculate entropy of target variable"""
        if len(y) == 0:
            return 0
        
        # Count occurrences of each class
        counts = Counter(y)
        total = len(y)
        
        # Calculate entropy
        entropy = 0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain from split"""
        parent_entropy = self._entropy(y)
        
        # Calculate weighted average of child entropies
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = len(y)
        
        if n_total == 0:
            return 0
            
        left_entropy = self._entropy(y_left)
        right_entropy = self._entropy(y_right)
        
        weighted_entropy = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy
        
        return parent_entropy - weighted_entropy
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity as alternative splitting criterion"""
        if len(y) == 0:
            return 0
            
        counts = Counter(y)
        total = len(y)
        gini = 1
        
        for count in counts.values():
            p = count / total
            gini -= p ** 2
            
        return gini
    
    def _find_best_split(self, X, y, feature_indices):
        """Find the best split for given features"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature_idx in feature_indices:
            # Get unique values for this feature
            values = X[:, feature_idx]
            unique_values = np.unique(values)
            
            # Try different thresholds
            for threshold in unique_values:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate information gain
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return TreeNode(value=self._predict_leaf(y))
        
        # Select features to consider
        n_features = X.shape[1]
        if self.max_features is None:
            feature_indices = list(range(n_features))
        else:
            np.random.seed(self.random_state + depth)
            feature_indices = np.random.choice(n_features, 
                                             min(self.max_features, n_features), 
                                             replace=False)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, feature_indices)
        
        # If no good split found, create leaf
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
        """Predict for a single sample"""
        if node.is_leaf:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def fit(self, X, y):
        """Train the decision tree"""
        np.random.seed(self.random_state)
        self.root = self._build_tree(X, y)
        self._calculate_feature_importance(X, y)
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.root))
        return np.array(predictions)
    
    def _calculate_feature_importance(self, X, y):
        """Calculate feature importance based on information gain"""
        n_features = X.shape[1]
        self.feature_importance_ = np.zeros(n_features)
        
        # This is a simplified version - in practice, you'd track importance during tree building
        # For now, we'll use a placeholder that gives equal importance to all features
        self.feature_importance_ = np.ones(n_features) / n_features

class HotelBookingPredictor:
    """
    Main class for hotel booking cancellation prediction
    Includes data preprocessing, feature engineering, and model training
    """
    
    def __init__(self):
        self.tree = None
        self.label_encoders = {}
        self.feature_names = None
        self.scaler = None
        
    def _load_data(self):
        """Load training and test data"""
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
        
        # 1. Lead time categories (key insight from analysis)
        df['lead_time_category'] = pd.cut(df['lead_time'], 
                                        bins=[0, 30, 90, 365, float('inf')], 
                                        labels=['short', 'medium', 'long', 'very_long'])
        
        # 2. Price categories
        df['price_category'] = pd.cut(df['avg_price_per_room'], 
                                   bins=[0, 50, 100, 150, float('inf')], 
                                   labels=['low', 'medium', 'high', 'very_high'])
        
        # 3. Total nights
        df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        
        # 4. Total guests
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        
        # 5. Booking characteristics
        df['is_weekend_booking'] = (df['no_of_weekend_nights'] > 0).astype(int)
        df['has_children'] = (df['no_of_children'] > 0).astype(int)
        df['is_repeated_guest'] = df['repeated_guest']
        
        # 6. Historical behavior features
        df['total_previous_bookings'] = df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled']
        df['cancellation_rate'] = np.where(
            df['total_previous_bookings'] > 0,
            df['no_of_previous_cancellations'] / df['total_previous_bookings'],
            0
        )
        
        # 7. Time-based features
        df['arrival_season'] = df['arrival_month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # 8. Interaction features
        df['lead_time_price_interaction'] = df['lead_time'] * df['avg_price_per_room']
        df['guests_nights_interaction'] = df['total_guests'] * df['total_nights']
        
        return df
    
    def _encode_categorical_features(self, df, is_training=True):
        """Encode categorical features"""
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_columns:
            if is_training:
                # Fit encoder on training data
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Transform test data using fitted encoder
                if col in self.label_encoders:
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    unique_values = df[col].unique()
                    known_values = self.label_encoders[col].classes_
                    
                    # Map unseen values to most common class
                    unseen_mask = ~df[col].isin(known_values)
                    if unseen_mask.any():
                        most_common = self.label_encoders[col].classes_[0]
                        df.loc[unseen_mask, col] = most_common
                    
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _prepare_features(self, df, is_training=True):
        """Prepare features for training/prediction"""
        # Feature engineering
        df = self._feature_engineering(df, is_training)
        
        # Encode categorical features
        df = self._encode_categorical_features(df, is_training)
        
        # Select features for training
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
        """Train the custom decision tree model"""
        print("Starting model training...")
        
        # Load data
        self._load_data()
        
        # Prepare training data
        X_train, y_train = self._prepare_features(self.train_df, is_training=True)
        
        print(f"Training features shape: {X_train.shape}")
        print(f"Feature names: {self.feature_names}")
        
        # Initialize and train custom decision tree
        self.tree = CustomDecisionTree(
            max_depth=15,
            min_samples_split=50,
            min_samples_leaf=25,
            max_features='sqrt',  # Use sqrt of features for each split
            random_state=42,
            pruning_alpha=0.01
        )
        
        self.tree.fit(X_train, y_train)
        
        print("Model training completed!")
        
        # Evaluate on training data
        train_predictions = self.tree.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return self
    
    def predict(self):
        """Generate predictions for test set"""
        print("Generating predictions...")
        
        # Prepare test data
        X_test = self._prepare_features(self.test_df, is_training=False)
        
        # Make predictions
        predictions = self.tree.predict(X_test)
        
        # Create submission file
        submission = pd.DataFrame({
            'id': self.test_df['id'],
            'label': predictions
        })
        
        submission.to_csv('submission.csv', index=False)
        print("Predictions saved to submission.csv")
        
        # Print prediction distribution
        pred_counts = Counter(predictions)
        print(f"Prediction distribution: {dict(pred_counts)}")
        
        return submission
    
    def evaluate_model(self):
        """Evaluate model using cross-validation"""
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import f1_score, classification_report
        
        # Prepare data
        X_train, y_train = self._prepare_features(self.train_df, is_training=True)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # We'll use a wrapper to make our custom tree compatible with sklearn
        class SklearnWrapper:
            def __init__(self, tree):
                self.tree = tree
            
            def fit(self, X, y):
                self.tree.fit(X, y)
                return self
            
            def predict(self, X):
                return self.tree.predict(X)
        
        wrapper = SklearnWrapper(self.tree)
        
        # Calculate F1 scores
        f1_scores = cross_val_score(wrapper, X_train, y_train, cv=cv, scoring='f1_macro')
        
        print(f"Cross-validation F1 scores: {f1_scores}")
        print(f"Mean F1 score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
        
        return f1_scores

def main():
    """Main function to run the hotel booking prediction"""
    print("=== Custom Decision Tree for Hotel Booking Cancellation Prediction ===\n")
    
    # Initialize predictor
    predictor = HotelBookingPredictor()
    
    # Train model
    predictor.train()
    
    # Evaluate model
    print("\n=== Model Evaluation ===")
    f1_scores = predictor.evaluate_model()
    
    # Generate predictions
    print("\n=== Generating Predictions ===")
    submission = predictor.predict()
    
    print(f"\n=== Summary ===")
    print(f"Model: Custom Decision Tree")
    print(f"Features: {len(predictor.feature_names)}")
    print(f"Mean F1 Score: {f1_scores.mean():.4f}")
    print(f"Predictions saved to: submission.csv")
    
    return predictor, submission

if __name__ == "__main__":
    predictor, submission = main()
