Overview: This dataset records some features with labels. Your goal is to train a model to predict if a customer buys a house. 
Training data: train.csv

- Each row is a data record with its id, 22 attributes, and a label.

- The last cell of each row (“label” column) is “class label” in integer, which is our classification target. 
  o This label indicates if a customer buys a house.
    1 indicates buy.

    0 indicates not buy.

- Training data includes training records with ground-truth class labels.

- Use the training data to train your solution.

  Testing data: test.csv

- Each row is a data record with its id and 22 attributes. The attributes are the same as those of train.csv, but the label is missing.

- Use your method to get your predicted labels of the testing records in test.csv and generate submission file which includes your predictions in the format of sample_submission.csv.

- We will obtain your solutions’ performance based on your predicted labels for testing data.

- Our evaluation is based on the Macro-F1 metric.
  o A detailed explanation of the calculation of the metrics: 

  ​    https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and- multilabel-classification

Tasks and Requirements

-    Develop three solutions, one for each dataset, to predict the class label of each data record.
-    You can develop any solutions, based on either the algorithms introduced in this subject or the methods beyond the course content.
-    Macro-F1 will be used as the evaluation metric.
-    Any programming language
     o Your code should be clean and well-documented (e.g., with sufficient comments)
-    You can use low-level third-party packages to facilitate your implementation.
-    Your implementation should involve sufficient technical details developed by yourselves.
     o DO NOT simply call ready-to-use classification models provided in existing packages, as a Blackbox, to finish the project.
-    You MAY use third-parity libraries. As long as your implementation involves reasonable algorithmic details for solving the problem, then it is fine. Unless it is too obvious, we will be very moderate when deciding if an implementation is solely based on Blackbox algorithms.



