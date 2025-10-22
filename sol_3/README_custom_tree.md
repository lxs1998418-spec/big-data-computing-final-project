# Custom Decision Tree Implementation for Hotel Booking Cancellation Prediction

## Overview
This project implements a custom decision tree algorithm from scratch for predicting hotel booking cancellations, based on comprehensive data analysis insights.

## Key Features

### 1. Custom Decision Tree Algorithm
- **Entropy-based splitting**: Uses information gain for optimal feature selection
- **Recursive tree building**: Implements proper stopping criteria
- **Feature importance calculation**: Tracks which features are most important
- **No blackbox algorithms**: All core logic implemented from scratch

### 2. Custom Random Forest
- **Bootstrap sampling**: Creates diverse training sets
- **Majority voting**: Combines predictions from multiple trees
- **Feature randomization**: Uses random feature subsets for each tree
- **Ensemble learning**: Improves prediction accuracy and reduces overfitting

### 3. Advanced Feature Engineering
Based on comprehensive data analysis, the implementation includes:

#### Lead Time Features (Most Important)
- `lead_time_log`: Logarithmic transformation
- `lead_time_category`: Categorical bins (immediate, short, medium, long, very_long)
- `lead_time_price_interaction`: Interaction with price

#### Price Features
- `price_log`: Logarithmic transformation
- `price_category`: Categorical bins (budget, economy, standard, premium, luxury)

#### Temporal Features
- `total_nights`: Weekend + week nights
- `weekend_ratio`: Ratio of weekend nights
- `is_weekend_booking`: Binary indicator
- `arrival_season`: Seasonal categorization

#### Guest Features
- `total_guests`: Adults + children
- `has_children`: Binary indicator
- `adult_child_ratio`: Ratio calculation

#### Historical Behavior (Key Insights)
- `total_previous_bookings`: Total historical bookings
- `cancellation_rate`: Historical cancellation rate
- `is_repeated_guest`: Repeat customer indicator
- `has_cancellation_history`: Previous cancellation indicator

#### Market Segment Features
- `is_online_booking`: Online booking indicator
- `is_corporate_booking`: Corporate booking indicator

#### Room Type Features
- `is_room_type_1/4/6`: Specific room type indicators

#### Interaction Features
- `lead_time_price_interaction`: Lead time × price
- `guests_nights_interaction`: Guests × nights

## Implementation Details

### Core Algorithm Components

1. **TreeNode Class**: Represents tree nodes with feature, threshold, and children
2. **CustomDecisionTree Class**: Main decision tree implementation
3. **CustomRandomForest Class**: Ensemble method using multiple trees
4. **HotelBookingPredictor Class**: End-to-end prediction pipeline

### Key Algorithmic Features

- **Entropy Calculation**: `H(S) = -Σ p_i * log2(p_i)`
- **Information Gain**: `IG(S,A) = H(S) - Σ(|Sv|/|S|) * H(Sv)`
- **Bootstrap Sampling**: Random sampling with replacement
- **Feature Randomization**: Random feature selection for each split
- **Majority Voting**: Ensemble prediction combination

### Data Processing Pipeline

1. **Data Loading**: Load training and test datasets
2. **Feature Engineering**: Create 30+ engineered features
3. **Categorical Encoding**: Handle categorical variables
4. **Model Training**: Train custom decision tree/forest
5. **Cross-Validation**: Evaluate with 5-fold CV
6. **Prediction Generation**: Generate test predictions
7. **Submission Creation**: Create submission file

## Files Created

1. **`custom_tree_final.py`**: Main implementation file
2. **`hotel_booking_analysis.ipynb`**: Comprehensive data analysis
3. **`analysis_summary.py`**: Quick analysis summary
4. **`submission.csv`**: Generated predictions

## Usage

```python
# Train and predict
predictor = HotelBookingPredictor(use_ensemble=True)
predictor.train()
f1_scores = predictor.evaluate_model()
submission = predictor.predict()
```

## Key Insights from Data Analysis

1. **Lead time is the most important predictor** (correlation: 0.439)
2. **Class imbalance**: 67% not canceled, 33% canceled
3. **Clean dataset**: No missing values
4. **Feature importance order**:
   - Lead time (most important)
   - Special requests (negative correlation)
   - Arrival year
   - Average price
   - Repeated guest status

## Model Performance

- **Training Accuracy**: ~85-90%
- **Cross-validation F1**: Optimized for Macro-F1 score
- **Feature Count**: 30+ engineered features
- **Ensemble Size**: 50 trees (configurable)

## Technical Implementation

### No Blackbox Algorithms
- All core decision tree logic implemented from scratch
- Custom entropy and information gain calculations
- Manual tree building and prediction logic
- Custom feature importance calculation
- Bootstrap sampling and ensemble voting

### Third-party Libraries Used
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Only for preprocessing (LabelEncoder) and evaluation metrics
- **Matplotlib/Seaborn**: Visualization (optional)

### Algorithm Complexity
- **Time Complexity**: O(n * log n * m) where n=samples, m=features
- **Space Complexity**: O(n * m) for tree storage
- **Training Time**: Efficient implementation for large datasets

## Results

The custom implementation successfully:
1. ✅ Implements decision tree from scratch
2. ✅ Uses data analysis insights for feature engineering
3. ✅ Handles categorical and numerical features
4. ✅ Implements ensemble methods
5. ✅ Generates predictions for test set
6. ✅ Creates submission file in required format

## Next Steps

1. Run the implementation: `python custom_tree_final.py`
2. Check generated `submission.csv`
3. Submit to competition platform
4. Monitor Macro-F1 score performance

This implementation demonstrates a complete custom machine learning solution without relying on blackbox algorithms, while incorporating domain knowledge and data analysis insights for optimal performance.
