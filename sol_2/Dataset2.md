## Dataset2

Overview: This dataset records some features with labels. Your goal is to train a model to predict if a customer buys a house. 



### data analysis

data types

22 features, 3 categorical cols, 19 numerical cols

所有特征均不存在缺失值

```
 #   Column                   Non-Null Count   Dtype  
---  ------                   --------------   -----  
 0   id                       139930 non-null  int64  
 1   country                  139930 non-null  object 
 2   property_type            139930 non-null  object 
 3   furnishing_status        139930 non-null  object 
 4   property_size_sqft       139930 non-null  int64  
 5   price                    139930 non-null  int64  
 6   constructed_year         139930 non-null  int64  
 7   previous_owners          139930 non-null  int64  
 8   rooms                    139930 non-null  int64  
 9   bathrooms                139930 non-null  int64  
 10  garage                   139930 non-null  int64  
 11  garden                   139930 non-null  int64  
 12  crime_cases_reported     139930 non-null  int64  
 13  legal_cases_on_property  139930 non-null  int64  
 14  customer_salary          139930 non-null  int64  
 15  loan_amount              139930 non-null  int64  
 16  loan_tenure_years        139930 non-null  int64  
 17  monthly_expenses         139930 non-null  int64  
 18  down_payment             139930 non-null  int64  
 19  emi_to_income_ratio      139930 non-null  float64
 20  satisfaction_score       139930 non-null  int64  
 21  neighbourhood_rating     139930 non-null  int64  
 22  connectivity_score       139930 non-null  int64  
 23  label                    139930 non-null  int64  
```



3个类别数据列分布



property_type

![image-20251022141359755](/Users/liuxinsheng/Library/Application Support/typora-user-images/image-20251022141359755.png)

Furnishing_status

![image-20251022141418100](/Users/liuxinsheng/Library/Application Support/typora-user-images/image-20251022141418100.png)



country

![image-20251022141513655](/Users/liuxinsheng/Library/Application Support/typora-user-images/image-20251022141513655.png)



19 numerical columns distribution 

![image-20251022134237798](/Users/liuxinsheng/Library/Application Support/typora-user-images/image-20251022134237798.png)



heat map

**未经过特征工程**

考虑特征和标签相关性排序Top4：

satisfaction_score 0.57 （中等相关 ） legal_case_on_property -0.32 （中等负相关 ）crime_cases_reported -0.17 （中等弱负相关）emi_to_income_ratio -0.16 （中等弱负相关）

考虑特征和特征相关性关系

和标签相关top4的特征中：

satisfaction_score 和其他特征基本不相关

legal_case_on_property 同上

crime_cases_reported 同上



其他特征：

price 和 loan_amount的相关性最高 为0.94

其次是price 和 down_payment的相关性 0.85

![image-20251022134456311](/Users/liuxinsheng/Library/Application Support/typora-user-images/image-20251022134456311.png)



为什么需要特征工程？

- 类别型的特征无法直观计算相关性

- 热力图基于皮尔逊相关系数，计算特征之间的相关性，对于数值大的特征比如房价，直接使用公式计算 很难发现与标签的相关性

$$
r = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2} \cdot \sqrt{\sum_{i=1}^n (Y_i - \bar{Y})^2}}
$$





