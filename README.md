# FIT_Test

Install requirements.txt first => "pip install -r requirements.txt"

1. Basic solution: use Adaboost, please execute => "python main.py"
=> predictions.csv is the result of predicting test.csv

2. Better solution: use xgboos, please execute => "python main_xgboost.py"
=> predictions_xgboost.csv is the result when using xgboost.

feature_importance is also printed, it seems "Product number" & "age" are
the most important two attributes affect whether exit or not.
