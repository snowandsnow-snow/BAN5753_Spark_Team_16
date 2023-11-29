# BAN5753_Spark_Team_16
# Mini-Project-2
The objective of this project is to identify clients who will subscribe (yes/no) for a term deposit.
## 1. Exploratory Data Analysis (EDA)
1. In the downloaded csv dataset, Fistly use "," to divide the data into different columns.
2. Replcae all "." in data or column name to "-" since the code reported error to "."
###  Missing data
There are no missing data but the categoty "unknown" existing in some varaibles
### Univariate patterns
For Numerical Variables: 
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/039924ba-b4ce-4af3-afdd-a231ca29f910)

* Except 'AGE' is near normal distributed, all other variables are not normal distributed.
* These variables will be grouped into two categorices according to their distribution:
1. PDAYS
2. EURIBOR3M
* Others will be normalized using 'StandardScaler'.

For Categorical Variables:
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/2a666dde-0586-4cb1-bd7d-25bf5a173585)

Some caterical variables which have too many categories but very unbalanced, they will be grouped into less categories:
1. education: college VS. non-college
2. month :special months (Mar. & Sep. & Otc. & Dec) VS. others
### Bivariate analysis of target versus categorical input variables
Compare the target variale (yes/no) for each category of each variable using bar plot.
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/48cbd492-1b09-4a84-9c8a-5fc1a8b684cb)

### Bivariate analysis of target versus numerical input variables
Compare the target variale (yes/no) for each variable using histogram.
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/aafdbc16-3819-4aa7-aa05-d8082e6e4c8d)

### Transform
Try log transformation for some varible but only duration eliminetd some skew problem. But the log(duration) resulted in bad model results so finally give up.
### Correlations-colinearity problem
Accoring to this guidline,collinarity problem exist in these pairs:
* Emp_var_rate VS. cons_price_idx
* Emp_var_rate VS.euribor3m
* Emp_var_rate VS. nr_employed
* euribor3m VS. cons_price_idx
* euribor3m VS. nr_employed
* Pdays VS. previous

Deal with these two variables to solve collinearity problem:
* Emp_var_rate: remove it in modeling building
* euribor3m: change it to categorical variabe including two groups
* Pdays : change it to categorical variabe including two groups 


## 2. Prepare Data for Machine Learning
### StringIndexer, OneHotEncoderEstimator, OneHotEncoderEstimator, LabelIndexer, StandardScaler
### K-means Clustering
4 clusters were divided using K means method.
## 3. Train/Test split
train: test = 0.8:0.2
## 4. Confusion Matrix
The confusion matrix was defeined in this section
## 5 Supervised Models
### Model # 1: Logistic Regression
* Accuracy-Logistic Regression:  0.903037667071689
* Test Area Under ROC-Logistic Regression: 0.92004
### Model # 2: Decision Tree
 * Accuracy-Decision Tree :   0.9087
 * Test Area Under ROC-Decision Tree: 0.6453
### Model # 3: Random Forest
* Accuracy-Random Forest :  0.8995
* Test Area Under ROC -Random Forest 0.9166
### Model # 4: Gradient-boosted tree
* Accuracy-Gradient-boosted tree :  0.9086
* Test Area Under ROC-Gradient-boosted tree 0.9307
###  Model # 5: Factorization machines classifier
* Accuracy-Factorization machines classifier :  0.9011
* Test Area Under ROC-Factorization machines classifier 0.9101
### Findings and benefits of the champion model
The best mode is gradient-boosted tree. The reason may be because its sampling method and we train them to correct each other's errors, they're capable of capturing complex patterns in the data.
## 6. Best Model Saving and Load for future use
We look at the simplest model with the highest AUC values as our model, since logistic regression gives us 92% AUC which is not much lower than GBT at 93%  and is easier to explain to stakeholders we select the logistic regression as our final model

## 7. Prescriptive recommendations
Based on the decriptive analysis, we have seen a higher subscribe rate among:

* Highly educated
* Single people
* Special months (Mar. & Sep. & Otc. & Dec)
To capitalize on that, we should focus advertising on that demographic with targeted advertising in special months (Mar. & Sep. & Otc. & Dec)s, highly educated and single people
