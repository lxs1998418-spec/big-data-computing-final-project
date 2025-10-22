Overview: This dataset records information about booking reservations. Your goal is to train a model to classify if a customer will cancel the hotel booking appointment or not based on the training set.
Training data: train.csv
-    Each row is a data record with its id, 17 attributes, and a label.
-    The last cell of each row (“label” column) is “class label” in integer, which is our classification target. 
o This label indicates whether the appointment is canceled or not. 
-    1 indicates canceled.
-    0 indicates not canceled.
-    Training data includes training records with ground-truth class labels.
-    Use the training data to train your solution.
     Testing data: test.csv
-    Each row is a data record with its id and 17 attributes. The attributes are the same as those of train.csv, but the label is missing.
-    Use your method to get your predicted labels of the testing records in test.csv and generate submission file which includes your predictions in the format of 
     sample_submission.csv.
-    We will obtain your solutions’ performance based on your predicted labels for testing data.
-    Our evaluation is based on the Macro-F1 metric.
-    After generating your predicted labels, you need to submit it to our online platform.
     Sample Submission: sample_submission.csv
     -    This file is a sample of your submission about the predictions of test.csv.
     -    This file includes two columns.
          o id: the identifier of each record in test.csv.
          o label: the prediction of this record.