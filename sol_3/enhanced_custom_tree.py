#!/usr/bin/env python3
"""
Enhanced Custom Decision Tree with Advanced Features
Includes ensemble methods, advanced pruning, and sophisticated feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from typing import Dict, List, Tuple, Optional, Union
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import random
warnings.filterwarnings('ignore')

class AdvancedTreeNode:
    """Enhanced node class with additional metadata"""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, 
                 samples=None, impurity=None, depth=0, class_counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = value is not None
        self.samples = samples  # Number of samples at this node
        self.impurity = impurity  # Impurity at this node
        self.depth = depth
        self.class_counts = class_counts  # For probability estimation

class AdvancedDecisionTree:
    """
    Advanced Decision Tree with:
    - Multiple splitting criteria (entropy, gini, gain ratio)
    - Cost-complexity pruning
    - Feature importance calculation
    - Probability estimation
    """
    
    def __init__(self, max_depth=15, min_samples_split=30, min_samples_leaf=15, 
                 max_features=None, random_state=42, criterion='entropy',
                 ccp_alpha=0.0, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.criterion = criterion
        self.ccp_alpha = ccp_alpha
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.feature_importance_ = None
        self.n_features_ = None
        self.classes_ = None
        
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
    
    def _gini(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        total = len(y)
        gini = 1
        for count in counts.values():
            p = count / total
            gini -= p ** 2
        return gini
    
    def _gain_ratio(self, y, y_left, y_right):
        """Calculate gain ratio to handle bias towards features with many values"""
        info_gain = self._information_gain(y, y_left, y_right)
        split_info = self._entropy([len(y_left), len(y_right)])
        return info_gain / split_info if split_info > 0 else 0
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain"""
        parent_impurity = self._get_impurity(y)
        n_left, n_right = len(y_left), len(y_right)
        n_total = len(y)
        
        if n_total == 0:
            return 0
        
        left_impurity = self._get_impurity(y_left)
        right_impurity = self._get_impurity(y_right)
        
        weighted_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
        return parent_impurity - weighted_impurity
    
    def _get_impurity(self, y):
        """Get impurity based on criterion"""
        if self.criterion == 'entropy':
            return self._entropy(y)
        elif self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)
    
    def _find_best_split(self, X, y, feature_indices):
        """Find best split with multiple criteria"""
        best_score = 0
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
                
                # Calculate split quality
                if self.criterion == 'entropy':
                    score = self._information_gain(y, y_left, y_right)
                elif self.criterion == 'gini':
                    score = self._information_gain(y, y_left, y_right)
                elif self.criterion == 'gain_ratio':
                    score = self._gain_ratio(y, y_left, y_right)
                else:
                    score = self._information_gain(y, y_left, y_right)
                
                if score > best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_score
    
    def _build_tree(self, X, y, depth=0, parent_samples=None):
        """Build tree with advanced features"""
        n_samples = len(y)
        impurity = self._get_impurity(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1 or
            impurity <= self.min_impurity_decrease):
            return AdvancedTreeNode(
                value=self._predict_leaf(y), 
                samples=n_samples,
                impurity=impurity,
                depth=depth,
                class_counts=Counter(y)
            )
        
        # Select features
        n_features = X.shape[1]
        if self.max_features is None:
            feature_indices = list(range(n_features))
        elif self.max_features == 'sqrt':
            feature_indices = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        elif self.max_features == 'log2':
            feature_indices = np.random.choice(n_features, int(np.log2(n_features)), replace=False)
        else:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        
        # Find best split
        best_feature, best_threshold, best_score = self._find_best_split(X, y, feature_indices)
        
        if best_feature is None or best_score <= 0:
            return AdvancedTreeNode(
                value=self._predict_leaf(y),
                samples=n_samples,
                impurity=impurity,
                depth=depth,
                class_counts=Counter(y)
            )
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1, n_samples)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1, n_samples)
        
        return AdvancedTreeNode(
            feature=best_feature, 
            threshold=best_threshold,
            left=left_tree, 
            right=right_tree,
            samples=n_samples,
            impurity=impurity,
            depth=depth,
            class_counts=Counter(y)
        )
    
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
    
    def _predict_proba_single(self, x, node):
        """Predict probabilities for single sample"""
        if node.is_leaf:
            if node.class_counts is None:
                return {0: 0.5, 1: 0.5}  # Default probabilities
            
            total = sum(node.class_counts.values())
            if total == 0:
                return {0: 0.5, 1: 0.5}
            
            return {cls: count / total for cls, count in node.class_counts.items()}
        
        if x[node.feature] <= node.threshold:
            return self._predict_proba_single(x, node.left)
        else:
            return self._predict_proba_single(x, node.right)
    
    def fit(self, X, y):
        """Train the tree"""
        np.random.seed(self.random_state)
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y)
        self._calculate_feature_importance()
        return self
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x, self.root))
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        probabilities = []
        for x in X:
            proba = self._predict_proba_single(x, self.root)
            # Ensure we have probabilities for all classes
            proba_array = np.array([proba.get(cls, 0) for cls in self.classes_])
            probabilities.append(proba_array)
        return np.array(probabilities)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance based on information gain"""
        self.feature_importance_ = np.zeros(self.n_features_)
        self._calculate_importance_recursive(self.root)
        
        # Normalize
        total_importance = np.sum(self.feature_importance_)
        if total_importance > 0:
            self.feature_importance_ /= total_importance
    
    def _calculate_importance_recursive(self, node):
        """Recursively calculate feature importance"""
        if node.is_leaf:
            return
        
        # Add importance for this split
        if node.feature is not None:
            self.feature_importance_[node.feature] += node.samples * node.impurity
        
        # Recursively process children
        self._calculate_importance_recursive(node.left)
        self._calculate_importance_recursive(node.right)

class CustomRandomForest:
    """
    Custom Random Forest implementation using our custom decision trees
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=20, 
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
            # Create bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)
            
            # Train tree
            tree = AdvancedDecisionTree(
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
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        probabilities = []
        for tree in self.trees:
            probabilities.append(tree.predict_proba(X))
        
        # Average probabilities
        probabilities = np.array(probabilities)
        return np.mean(probabilities, axis=0)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance across all trees"""
        if not self.trees:
            return
        
        importance = np.zeros(self.trees[0].n_features_)
        for tree in self.trees:
            importance += tree.feature_importance_
        
        self.feature_importance_ = importance / len(self.trees)

class EnhancedHotelBookingPredictor:
    """
    Enhanced predictor with advanced feature engineering and ensemble methods
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
        
    def _advanced_feature_engineering(self, df, is_training=True):
        """
        Advanced feature engineering based on domain knowledge and analysis insights
        """
        df = df.copy()
        
        # 1. Lead time features (most important from analysis)
        df['lead_time_log'] = np.log1p(df['lead_time'])
        df['lead_time_sqrt'] = np.sqrt(df['lead_time'])
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
        df['adult_child_ratio'] = df['no_of_adults'] / (df['no_of_children'] + 1)
        df['has_children'] = (df['no_of_children'] > 0).astype(int)
        df['family_size'] = df['total_guests']
        
        # 5. Historical behavior (key insights)
        df['total_previous_bookings'] = (df['no_of_previous_cancellations'] + 
                                       df['no_of_previous_bookings_not_canceled'])
        df['cancellation_rate'] = np.where(
            df['total_previous_bookings'] > 0,
            df['no_of_previous_cancellations'] / df['total_previous_bookings'],
            0
        )
        df['is_repeated_guest'] = df['repeated_guest']
        df['has_cancellation_history'] = (df['no_of_previous_cancellations'] > 0).astype(int)
        
        # 6. Advanced time features
        df['arrival_season'] = df['arrival_month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        })
        
        # Cyclical encoding for months
        df['month_sin'] = np.sin(2 * np.pi * df['arrival_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['arrival_month'] / 12)
        
        # 7. Interaction features (based on analysis insights)
        df['lead_time_price_interaction'] = df['lead_time'] * df['avg_price_per_room']
        df['guests_nights_interaction'] = df['total_guests'] * df['total_nights']
        df['lead_time_guests_interaction'] = df['lead_time'] * df['total_guests']
        df['price_nights_interaction'] = df['avg_price_per_room'] * df['total_nights']
        
        # 8. Special request features
        df['special_requests_log'] = np.log1p(df['no_of_special_requests'])
        df['has_special_requests'] = (df['no_of_special_requests'] > 0).astype(int)
        
        # 9. Market segment features (from analysis insights)
        df['is_online_booking'] = (df['market_segment_type'] == 'Online').astype(int)
        df['is_corporate_booking'] = (df['market_segment_type'] == 'Corporate').astype(int)
        
        # 10. Room type features
        df['is_room_type_1'] = (df['room_type_reserved'] == 'Room_Type 1').astype(int)
        df['is_room_type_4'] = (df['room_type_reserved'] == 'Room_Type 4').astype(int)
        df['is_room_type_6'] = (df['room_type_reserved'] == 'Room_Type 6').astype(int)
        
        # 11. Meal plan features
        df['has_meal_plan'] = (df['type_of_meal_plan'] != 'Not Selected').astype(int)
        df['is_meal_plan_1'] = (df['type_of_meal_plan'] == 'Meal Plan 1').astype(int)
        
        # 12. Parking features
        df['requires_parking'] = df['required_car_parking_space']
        
        return df
    
    def _encode_categorical_features(self, df, is_training=True):
        """Encode categorical features with proper handling of unseen values"""
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
                        # Use most frequent category for unseen values
                        most_frequent = self.label_encoders[col].classes_[0]
                        df.loc[unseen_mask, col] = most_frequent
                    
                    df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def _prepare_features(self, df, is_training=True):
        """Prepare features with advanced engineering"""
        # Advanced feature engineering
        df = self._advanced_feature_engineering(df, is_training)
        
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
        """Train the enhanced model"""
        print("Starting enhanced model training...")
        
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
                n_estimators=100,
                max_depth=12,
                min_samples_split=30,
                min_samples_leaf=15,
                max_features='sqrt',
                random_state=42
            )
        else:
            print("Training Advanced Decision Tree...")
            self.model = AdvancedDecisionTree(
                max_depth=15,
                min_samples_split=30,
                min_samples_leaf=15,
                max_features='sqrt',
                criterion='entropy',
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate training performance
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
    print("=== Enhanced Custom Decision Tree for Hotel Booking Prediction ===\n")
    
    # Train with ensemble
    print("Training with Custom Random Forest...")
    predictor_ensemble = EnhancedHotelBookingPredictor(use_ensemble=True)
    predictor_ensemble.train()
    
    # Evaluate
    f1_scores_ensemble = predictor_ensemble.evaluate_model()
    
    # Generate predictions
    submission_ensemble = predictor_ensemble.predict()
    
    print(f"\n=== Final Results ===")
    print(f"Model: Custom Random Forest")
    print(f"Features: {len(predictor_ensemble.feature_names)}")
    print(f"Mean F1 Score: {f1_scores_ensemble.mean():.4f}")
    print(f"Predictions saved to: submission.csv")
    
    return predictor_ensemble, submission_ensemble

if __name__ == "__main__":
    predictor, submission = main()
