# BAN5753_Spark_Team_16
# Mini-Project-2
The objective of this project is to identify clients who will subscribe (yes/no) for a term deposit.
## 1. Exploratory Data Analysis (EDA)
1. In the downloaded csv dataset, Fistly use "," to divide the data into different columns.
2. Replcae all "." in data or column name to "-" since the code reported error to "."
###  Missing Data
There are no missing data but the categoty "unknown" existing in some varaibles.
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/fa62fff2-f614-4f37-8059-c7b249a98648)

### Univariate Patterns - Numerical Variables
For Numerical Variables, the distribution for each variable is shown below: 
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/039924ba-b4ce-4af3-afdd-a231ca29f910)

* Except 'AGE' is near normal distributed, all other variables are not normal distributed.
* These variables will be grouped into two categorices according to their distribution:
1. PDAYS
2. EURIBOR3M
* Others will be normalized using 'StandardScaler'.
### Univariate Patterns - Categorical Variables
For Categorical Variables, the distribution for each variable is shown below:
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/2a666dde-0586-4cb1-bd7d-25bf5a173585)

Some caterical variables which have too many categories but very unbalanced, they will be grouped into less categories:
1. education: college VS. non-college
2. month :special months (Mar. & Sep. & Otc. & Dec) VS. others
   
### Bivariate Analysis of Target Versus Categorical Input Variables
Compare the target variale (yes/no) for each category of each variable using bar plot:
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/48cbd492-1b09-4a84-9c8a-5fc1a8b684cb)
From these we could see which categories impact users decision to subscribe, we are able to dwell deeper into building  profiles of our customers.
We see that some variables can be looked into to provide insights since the subscribe rate for yes are very unbalanced:
1. education_udf
2. maritial status
3. month

### Bivariate Analysis of Target Versus Numerical Input Variables
Compare the target variale (yes/no) for each variable using histogram:
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/aafdbc16-3819-4aa7-aa05-d8082e6e4c8d)
We see that for every one of the features, there is a class imbalance in each histogram, we see that there is more data for y = no as compared to y = yes.
Because of this we will have to look for evaluation metrics and machine learning models which could take this into consideration


### Transforming
Try log transformation for some varible but only duration eliminetd some skew problem. But the log(duration) resulted in bad model results so finally give up. The another reason for giving up log transdormation is beacuse we will use standard scaler.
### Correlations-colinearity Problem
The collelation reuslts are shown below.
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/b36644a1-17d5-4f14-aa5d-74121e9330bc)

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
### Variable preparation
The following technologies were used:

1. StringIndexer
2. OneHotEncoderEstimator
3. LabelIndexer
4. StandardScaler
### K-means Clustering
4 clusters were divided using K means method.
## 3. Train/Test split
train: test = 0.8:0.2
## 4. Confusion Matrix
The confusion matrix was defeined in this section. The confusion matrix for the five models are shown below:
![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/26d90f2b-0fb7-4d04-b59a-1150e402507f)

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
### Findings and Benefits of the Champion Model
The best performance model is gradient-boosted tree (GBT) accoring to AUC. The reason may be because its sampling method and we train them to correct each other's errors, they're capable of capturing complex patterns in the data. However, the logistic regression gives us 92% AUC which is not much lower than GBT at 93% but simpler than GBT model.
## 6. Best Model Saving and Load for future use
We look at the simplest model with the highest AUC values as our model, since logistic regression gives us 92% AUC which is not much lower than GBT at 93%  and is easier to explain to stakeholders we select the logistic regression as our final model

## 7. Prescriptive Recommendations
Based on the decriptive analysis (EDA), we have seen a higher subscribe rate among:

* Highly educated
* Single people
* Special months (Mar. & Sep. & Otc. & Dec)
To capitalize on that, we should focus advertising on that demographic with targeted advertising:
1. For highly educated: Target marketing strategies that can help users gain discounts on their future investments, highly educated users also would be tech savy so partnering with tech firms to offer deals could be efficient
2. For single users: Create campaigns that appeal to their lifestlye, this could include deals adding value to their individual portfolio
3. For certain months :Plan and time marketing campaigns to coincide with March , September , October and December, this could look like seasonal offers or promotions that align with festivals ![image](https://github.com/snowandsnow-snow/BAN5753_Spark_Team_16/assets/63618493/35836488-867e-4626-a0ce-44b508f1ac1e)

