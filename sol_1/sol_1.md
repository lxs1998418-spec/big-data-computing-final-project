Dataset 1 
Overview: This dataset records information about some earthquake incidents. Your goal is to train a model to predict the alert level to be raised based on the training set.
Training data: train.csv

-    Each row is a data record with its id, 5 attributes, and a label.
-    The last cell of each row (“label” column) is “class label” in integer, which is our classification target. 
     o This label indicates the alert level (from 0 to 3).
-    Training data includes training records with ground-truth class labels.
-    Use the training data to train your solution.
     Testing data: test.csv
-    Each row is a data record with its id and 5 attributes. The attributes are the same as those of train.csv, but the label is missing.
-    Use your method to get your predicted labels of the testing records in test.csv and generate submission file which includes your predictions in the format of sample_submission.csv.
-    We will obtain your solutions’ performance based on your predicted labels for testing data.
-    Our evaluation is based on the Macro-F1 metric.



Tasks and Requirements

-    Develop three solutions, one for each dataset, to predict the class label of each data record.
-    You can develop any solutions, based on either the algorithms introduced in this subject or the methods beyond the course content.
-    Macro-F1 will be used as the evaluation metric.
-    Any programming language
     o Your code should be clean and well-documented (e.g., with sufficient comments)
-    You can use low-level third-party packages to facilitate your implementation.
-    Your implementation should involve sufficient technical details developed by yourselves.
     o DO NOT simply call ready-to-use classification models provided in existing packages, as a Blackbox, to finish the project.