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

## 特征与标签关系分析

### 数值特征与标签相关性 (按绝对值排序)

| 排名 | 特征 | 相关系数 | 解释 |
|------|------|----------|------|
| 1 | lead_time | 0.4387 | **强正相关** - 提前预订时间越长，取消率越高 |
| 2 | no_of_special_requests | 0.2544 | **强负相关** - 特殊请求越多，取消率越低 |
| 3 | arrival_year | 0.1809 | 中等正相关 - 年份较新的预订取消率较高 |
| 4 | avg_price_per_room | 0.1420 | 中等正相关 - 价格越高，取消率越高 |
| 5 | repeated_guest | 0.1090 | 中等负相关 - 重复客户取消率较低 |
| 6 | no_of_week_nights | 0.0962 | 弱正相关 - 工作日住宿越多，取消率越高 |
| 7 | required_car_parking_space | 0.0856 | 弱正相关 - 需要停车位的客户取消率较高 |
| 8 | no_of_adults | 0.0789 | 弱正相关 - 成人数量越多，取消率越高 |

### 分类特征与标签关系 (卡方检验)

| 特征 | 卡方值 | p值 | 显著性 |
|------|--------|-----|--------|
| market_segment_type | 568.0423 | 0.000000 | **极显著** |
| type_of_meal_plan | 175.1280 | 0.000000 | **极显著** |
| room_type_reserved | 39.6051 | 0.000001 | **极显著** |

所有分类特征都与标签有极显著的关系。

## 特征与特征关系分析

### 高相关性特征对
- `repeated_guest` - `no_of_previous_bookings_not_canceled`: 0.551
  - 重复客户与历史未取消预订次数呈强正相关，符合业务逻辑

### 其他特征间相关性
大部分特征间的相关性较弱（< 0.5），说明特征相对独立，有利于模型训练。

## 深度分析结果

### 1. 提前预订时间 (lead_time) 分析

| 时间段 | 预订数量 | 取消率 | 分析 |
|--------|----------|--------|------|
| 0-7天 | 3,130 | 10.19% | 短期预订取消率最低 |
| 7-30天 | 4,650 | 18.77% | 中期预订取消率中等 |
| 30-90天 | 7,468 | 25.50% | 长期预订取消率较高 |
| 90+天 | 9,087 | 55.87% | **超长期预订取消率最高** |

**关键发现**: 提前预订时间与取消率呈明显的正相关关系，提前预订时间越长，取消率越高。

### 2. 房间价格 (avg_price_per_room) 分析

| 价格区间 | 预订数量 | 取消率 | 分析 |
|----------|----------|--------|------|
| 0-50 | 262 | 9.92% | 低价房间取消率最低 |
| 50-100 | 12,601 | 26.84% | 中低价房间取消率中等 |
| 100-150 | 9,967 | 42.06% | **中价房间取消率最高** |
| 150-200 | 1,881 | 33.01% | 中高价房间取消率较高 |
| 200+ | 320 | 49.38% | 高价房间取消率很高 |

**关键发现**: 价格与取消率的关系呈现U型曲线，中价房间(100-150)取消率最高。

### 3. 特殊请求数量 (no_of_special_requests) 分析

| 特殊请求数 | 预订数量 | 取消率 | 分析 |
|------------|----------|--------|------|
| 0 | 13,920 | 43.41% | **无特殊请求取消率最高** |
| 1 | 7,983 | 23.98% | 1个特殊请求取消率中等 |
| 2 | 2,993 | 14.27% | 2个特殊请求取消率较低 |
| 3+ | 521 | 0.00% | **多个特殊请求取消率为0** |

**关键发现**: 特殊请求数量与取消率呈强负相关，特殊请求越多，客户越不容易取消。

### 4. 客户历史行为分析

#### 重复客户分析
- **非重复客户**: 24,760个预订，取消率33.82%
- **重复客户**: 657个预订，取消率1.52%

**关键发现**: 重复客户的取消率极低，是重要的正面指标。

#### 历史取消行为分析
- **无历史取消**: 25,181个预订，取消率33.25%
- **有历史取消**: 236个预订，取消率显著降低

**关键发现**: 有历史取消记录的客户，当前预订的取消率反而较低。

## 业务洞察与建议

### 1. 高风险客户识别
- **提前预订时间超过90天的客户** - 取消率高达55.87%
- **价格在100-150区间的客户** - 取消率高达42.06%
- **无特殊请求的客户** - 取消率高达43.41%

### 2. 低风险客户识别
- **重复客户** - 取消率仅1.52%
- **有多个特殊请求的客户** - 取消率为0%
- **提前预订时间在7天内的客户** - 取消率仅10.19%

### 3. 模型构建建议
1. **重要特征排序**:
   - lead_time (提前预订时间)
   - no_of_special_requests (特殊请求数量)
   - market_segment_type (市场细分类型)
   - avg_price_per_room (房间价格)
   - repeated_guest (重复客户)

2. **特征工程建议**:
   - 将lead_time分组处理
   - 创建价格区间特征
   - 考虑客户历史行为的组合特征
   - 特殊请求数量可以作为强正面指标

3. **模型选择建议**:
   - 考虑使用树模型（如XGBoost、LightGBM）处理非线性关系
   - 使用集成方法提高预测准确性
   - 重点关注Macro-F1指标优化

### 4. 业务策略建议
1. **风险管理**:
   - 对提前预订超过90天的客户收取更高押金
   - 对无特殊请求的客户进行额外确认
   - 对中价房间客户提供更多服务保障

2. **客户维护**:
   - 重点维护重复客户关系
   - 鼓励客户提出特殊请求
   - 对短期预订客户提供快速确认服务

## 结论

通过深入的数据分析，我们发现了多个与酒店预订取消行为密切相关的关键因素：

1. **提前预订时间**是最重要的预测因子，与取消率呈强正相关
2. **特殊请求数量**是重要的正面指标，与取消率呈强负相关  
3. **客户历史行为**对当前取消行为有重要影响
4. **价格因素**呈现复杂的非线性关系
5. **市场细分和房间类型**等分类特征也有显著影响

这些发现为构建高精度的预测模型提供了重要的数据基础，同时也为酒店业务策略制定提供了有价值的洞察。
