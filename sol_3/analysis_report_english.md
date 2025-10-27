# Hotel Booking Cancellation Prediction - Data Analysis Report

## Project Overview

This project analyzes hotel booking dataset to understand feature-feature relationships and feature-label relationships (booking cancellation), providing data insights for subsequent machine learning model development.

## Dataset Overview

- **Training samples**: 25,417
- **Test samples**: 10,858  
- **Number of features**: 17 (excluding id and label)
- **Numerical features**: 14
- **Categorical features**: 3

### Label Distribution
- **Not Cancelled (0)**: 17,034 (67.02%)
- **Cancelled (1)**: 8,383 (32.98%)

The dataset is relatively balanced with a cancellation rate of approximately 33%, suitable for binary classification prediction.

## Feature Analysis

### Numerical Features (14)
1. `no_of_adults` - Number of adults
2. `no_of_children` - Number of children  
3. `no_of_weekend_nights` - Number of weekend nights
4. `no_of_week_nights` - Number of week nights
5. `required_car_parking_space` - Whether car parking space is required
6. `lead_time` - Lead time (days between booking and arrival)
7. `arrival_year` - Year of arrival
8. `arrival_month` - Month of arrival
9. `arrival_date` - Date of arrival
10. `repeated_guest` - Whether it's a repeated guest
11. `no_of_previous_cancellations` - Number of previous cancellations
12. `no_of_previous_bookings_not_canceled` - Number of previous bookings not cancelled
13. `avg_price_per_room` - Average price per room
14. `no_of_special_requests` - Number of special requests

### Categorical Features (3)
1. `type_of_meal_plan` - Type of meal plan
2. `room_type_reserved` - Type of room reserved
3. `market_segment_type` - Market segment type

## Feature-Label Relationship Analysis

### Numerical Features Correlation with Label (Sorted by Absolute Value)

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|----------------|
| 1 | lead_time | 0.4387 | **Strong positive correlation** - Longer lead time, higher cancellation rate |
| 2 | no_of_special_requests | 0.2544 | **Strong negative correlation** - More special requests, lower cancellation rate |
| 3 | arrival_year | 0.1809 | Moderate positive correlation - Newer year bookings have higher cancellation rate |
| 4 | avg_price_per_room | 0.1420 | Moderate positive correlation - Higher price, higher cancellation rate |
| 5 | repeated_guest | 0.1090 | Moderate negative correlation - Repeated guests have lower cancellation rate |
| 6 | no_of_week_nights | 0.0962 | Weak positive correlation - More week nights, higher cancellation rate |
| 7 | required_car_parking_space | 0.0856 | Weak positive correlation - Customers requiring parking have higher cancellation rate |
| 8 | no_of_adults | 0.0789 | Weak positive correlation - More adults, higher cancellation rate |

### Categorical Features Relationship with Label (Chi-square Test)

| Feature | Chi-square Value | p-value | Significance |
|---------|------------------|---------|--------------|
| market_segment_type | 568.0423 | 0.000000 | **Highly significant** |
| type_of_meal_plan | 175.1280 | 0.000000 | **Highly significant** |
| room_type_reserved | 39.6051 | 0.000001 | **Highly significant** |

All categorical features have highly significant relationships with the label.

## Feature-Feature Relationship Analysis

### Highly Correlated Feature Pairs
- `repeated_guest` - `no_of_previous_bookings_not_canceled`: 0.551
  - Strong positive correlation between repeated guests and previous non-cancelled bookings, which aligns with business logic

### Other Feature Correlations
Most feature pairs have weak correlations (< 0.5), indicating relatively independent features, which is beneficial for model training.

## In-depth Analysis Results

### 1. Lead Time Analysis

| Time Period | Booking Count | Cancellation Rate | Analysis |
|-------------|---------------|-------------------|----------|
| 0-7 days | 3,130 | 10.19% | Short-term bookings have lowest cancellation rate |
| 7-30 days | 4,650 | 18.77% | Medium-term bookings have moderate cancellation rate |
| 30-90 days | 7,468 | 25.50% | Long-term bookings have higher cancellation rate |
| 90+ days | 9,087 | 55.87% | **Very long-term bookings have highest cancellation rate** |

**Key Finding**: Lead time shows a clear positive correlation with cancellation rate - the longer the lead time, the higher the cancellation rate.

### 2. Room Price Analysis

| Price Range | Booking Count | Cancellation Rate | Analysis |
|-------------|---------------|-------------------|----------|
| 0-50 | 262 | 9.92% | Low-price rooms have lowest cancellation rate |
| 50-100 | 12,601 | 26.84% | Low-medium price rooms have moderate cancellation rate |
| 100-150 | 9,967 | 42.06% | **Medium-price rooms have highest cancellation rate** |
| 150-200 | 1,881 | 33.01% | Medium-high price rooms have higher cancellation rate |
| 200+ | 320 | 49.38% | High-price rooms have very high cancellation rate |

**Key Finding**: Price shows a U-shaped relationship with cancellation rate, with medium-price rooms (100-150) having the highest cancellation rate.

### 3. Special Requests Analysis

| Special Requests | Booking Count | Cancellation Rate | Analysis |
|------------------|---------------|-------------------|----------|
| 0 | 13,920 | 43.41% | **No special requests have highest cancellation rate** |
| 1 | 7,983 | 23.98% | 1 special request has moderate cancellation rate |
| 2 | 2,993 | 14.27% | 2 special requests have lower cancellation rate |
| 3+ | 521 | 0.00% | **Multiple special requests have zero cancellation rate** |

**Key Finding**: Special requests show a strong negative correlation with cancellation rate - the more special requests, the less likely customers are to cancel.

### 4. Customer Historical Behavior Analysis

#### Repeated Guest Analysis
- **Non-repeated guests**: 24,760 bookings, 33.82% cancellation rate
- **Repeated guests**: 657 bookings, 1.52% cancellation rate

**Key Finding**: Repeated guests have extremely low cancellation rates, making this an important positive indicator.

#### Historical Cancellation Behavior Analysis
- **No previous cancellations**: 25,181 bookings, 33.25% cancellation rate
- **With previous cancellations**: 236 bookings, significantly lower cancellation rate

**Key Finding**: Customers with previous cancellation records actually have lower cancellation rates for current bookings.

## Business Insights and Recommendations

### 1. High-Risk Customer Identification
- **Customers with lead time over 90 days** - 55.87% cancellation rate
- **Customers in 100-150 price range** - 42.06% cancellation rate
- **Customers with no special requests** - 43.41% cancellation rate

### 2. Low-Risk Customer Identification
- **Repeated guests** - Only 1.52% cancellation rate
- **Customers with multiple special requests** - 0% cancellation rate
- **Customers with lead time within 7 days** - Only 10.19% cancellation rate

### 3. Model Building Recommendations
1. **Important Feature Ranking**:
   - lead_time (Lead time)
   - no_of_special_requests (Special requests)
   - market_segment_type (Market segment type)
   - avg_price_per_room (Room price)
   - repeated_guest (Repeated guest)

2. **Feature Engineering Suggestions**:
   - Group lead_time into bins
   - Create price range features
   - Consider customer historical behavior combination features
   - Use special requests as a strong positive indicator

3. **Model Selection Recommendations**:
   - Consider tree models (XGBoost, LightGBM) for handling non-linear relationships
   - Use ensemble methods to improve prediction accuracy
   - Focus on Macro-F1 metric optimization

### 4. Business Strategy Recommendations
1. **Risk Management**:
   - Charge higher deposits for customers booking more than 90 days in advance
   - Provide additional confirmation for customers with no special requests
   - Offer more service guarantees for medium-price room customers

2. **Customer Retention**:
   - Focus on maintaining relationships with repeated guests
   - Encourage customers to make special requests
   - Provide quick confirmation services for short-term booking customers

## Conclusion

Through in-depth data analysis, we have identified multiple key factors closely related to hotel booking cancellation behavior:

1. **Lead time** is the most important predictor, showing strong positive correlation with cancellation rate
2. **Special requests** is an important positive indicator, showing strong negative correlation with cancellation rate  
3. **Customer historical behavior** significantly affects current cancellation behavior
4. **Price factors** show complex non-linear relationships
5. **Market segment and room type** categorical features also have significant impacts

These findings provide important data foundation for building high-accuracy prediction models and offer valuable insights for hotel business strategy development.
