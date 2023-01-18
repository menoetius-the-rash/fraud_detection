# Online payments fraud detection
Fraud detection in online payments

## Description

The dataset used for this was taken from [Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
Further research and XGBoost was ended up being chosen to perform the fraud classification in online payments

### Environment

The specifications of the machine used to perform the code has 11th Gen Intel i9-11950H, 64GB ram, 2TB SSD storage, and a NVIDIA GeForce RTX 3080 Laptop GPU. Anaconda version 22.9.0 was used to install Jupyter Notebook version 6.4.12 with a Python version 3.9.12 use. The python packages imported and installed were time, numpy, matplotlib, seaborn, xgboost, imblearn, sklearn. These packages and environment information allow the reader to reproduce the process and generate the same results.

### Exploratory Data Analysis

The dataset was loaded into a dataframe and viewed to check the features that are to be processed.

![image](https://user-images.githubusercontent.com/62455043/213055246-c4bf701d-ea10-4e0d-8052-c4961fa37b04.png)

Summary of the table
The table confirms that columns nameOrig and nameDest have been input in such a way that they do not form any point to any person that can be identifiable based on the value.

![image](https://user-images.githubusercontent.com/62455043/213055240-fdf7cd40-41eb-406c-892a-e69fa38a6462.png)

Bar chart - breakdown of online transaction types
A bar chart was generated to understand the different types of online transactions executed in the dataset. It can be noticed that the majority of the transactions were CASH_OUT, PAYMENT, CASH_IN and TRANSFER. DEBIT transactions ranked the lowest with only 41,432 transactions.

![image](https://user-images.githubusercontent.com/62455043/213055228-90d0e9f5-e337-4e0c-9112-5e93c56bd54c.png)

Correlation matrix of features
A correlation matrix was then generated to understand the correlation coefficients for the different variables to be processed in the dataset. The matrix is shown in a form of a heatmap to help identify pairs that have high correlation. From visually viewing the matrix, it is noticed that oldbalanceDest and newbalanceDest columns have a very high correlation coefficient of 0.98. With amount and newbalanceDest having a 0.46 correlation coefficient score. Finally with amount and oldbalanceDest having a 0.29 correlation coefficient score.

![image](https://user-images.githubusercontent.com/62455043/213055218-47928b99-b4b4-4bae-b917-8798301806af.png)

Bar chart - breakdown of dataset non-fraud against fraud
The dataset was further broken down on what was identified as non-fraud and fraud. A huge imbalance between non-fraud and fraud was identified with 6354407 non-fraud transactions against 8213 fraud transactions. As there is a skewed class distribution, it is noted to perform techniques that relate to under sampling in order to balance class distribution between fraud and non-fraud.
 
![image](https://user-images.githubusercontent.com/62455043/213055200-ab182317-a4d8-4aed-8f52-8f75a12d2558.png)

Pie chart - breakdown of fraud online transaction types
With a focus on fraud transactions and the use of a pie chart, that the fraud transactions noted in the dataset are between TRANSFER and CASH_OUT.  

![image](https://user-images.githubusercontent.com/62455043/213055185-25d87a83-3783-43e6-8fd0-872a0c056d31.png)

Table of variance inflation factore (VIF)
A function was created in order to generate a VIF score for each variable. This is done to ensure that the measure can be seen on how much a variable can affect the standard error in the regression. From this table, it is seen that newbalanceOrig, oldbalanceOrg, newbalanceDest, oldbalanceDest highly affect the analysis from a regression perspective. This will be noted when regression analysis is performed.
 
![image](https://user-images.githubusercontent.com/62455043/213055166-8030a4a3-8454-4f53-ae4b-5b90c28d39de.png)

Top 10 recipients of fraud transactions
A ranking of top 10 was generated for the most active fraudulent users. It is noticed that there are users with greater than 100 transactions. A deeper dive is done into this to understand if there are any correlation between multiple transactions performed against money gained. Essentially, these were the most active fraud users.

![image](https://user-images.githubusercontent.com/62455043/213055142-f36c0fe0-c556-480d-a21e-af0181261b51.png)

Top 10 Total money gained per recipients on one transaction
A top 10 ranking of total money earned based on the user is performed on one transaction. It is noticed here when comparing both tables from Fig. 7 and 8, that there is no correlation between total money earned on one transaction and total number of transactions performed. 

![image](https://user-images.githubusercontent.com/62455043/213055124-58fedb75-1a12-4a68-907f-6d41c50fbc90.png)

Total money lost per victim on one fraud transaction
The table shows the amount a victim lost due to fraud in one transaction. What can be realized here is that the top 10 majority have had their accounts cleared to a balance of zero.
Taking this into account, the average fraud transaction was 124,294.73.

### Implementation & Methodology

As the dataset for the analysis has been chosen, the data requires to be preprocessed.

First, the data will be removed from any missing data fields. The reason for this is that since there are 6,362,620 rows in our dataset, removing rows that contain missing does not affect the end result of the analysis. Reviewing the current variables that are available, nameOrig  and nameDest were removed due to the fact that they did not add value to the analysis. 

Second, the categorical variable type was then label encoded. Once label encoded the dataset was transformed with additional variables type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT and type_TRANSER. Once this was done, other variables were created which were called DestVar and OrigVar. These two variables were created in order to calculate the variance of cashflow between accounts. A correlation matrix was then generated to analyze how the newly generated variables interact with the other variables.

![image](https://user-images.githubusercontent.com/62455043/213055801-aace42b5-7e63-449a-9535-4a53b9b9d69b.png)

Correlation matrix of features
It was noted that the variables had a high correlation, DestVar and amount, oldbalanceOrg and newbalanceOrig, oldbalanceDest and newbalanceDest. The variables DestVar, oldbalanceOrg, oldbalanceDest were removed due to the high correlation.

![image](https://user-images.githubusercontent.com/62455043/213055758-20dea929-28c0-497f-a5e4-b67319aa430f.png)

Confusion Matrix features after removal
A correlation matrix was generated again to review the interaction after removing the variables. As there are no correlation scores higher than 0.6 for both positive and negative correlation.

Third, as it was mentioned in CA1 that there is an imbalance in the variable classification of fraud, random undersampling was performed to the data in order to bring the data into equal distribution of classes. 

Fourth, the data was then split into 80% training data and 20% test data. XGBoost was then performed on the training data and after this, a prediction was performed on the test data.

As per KDD methodology, in the next section the results and evaluation will cover information retrieved from the data in which XGBoost was performed.

### Results & Evaluation

In order to evaluate the results from XGBoost, the model was used to generate a confusion matrix, a receiver operating characteristic (ROC) curve plot with area under curve (AUC) value, a classification report, accuracy, sensitivity, precision, and specificity scores were generated. 

![image](https://user-images.githubusercontent.com/62455043/213055672-f5b08ab4-a33a-46d4-a01e-b8281129764c.png)

Confusion Matrix of XGBoost model
As per the confusion matrix, it can be seen that the model is quite accurate. The true positive (TP) and true negatives (TN) have a high score and false positive (FP) and false negative (FN) both have a very low score. This is ideal in the business case as a higher detection rate of fraud is better.

![image](https://user-images.githubusercontent.com/62455043/213055643-cea82bbe-709d-4af7-ae12-5a23e64b9b51.png)

ROC Curve plot
A ROC curve plot was then generated along with an AUC score of 0.9945. With a lower FP and FN values, the curve becomes more sharp and closer to 1 at its threshold. With an AUC score of 0.9945, this shows the model has a high percentage of correct predictions.

![image](https://user-images.githubusercontent.com/62455043/213055613-35d88921-08ee-4926-b1c6-f61789342dcd.png)

Classification Report
The classification correlates to the results shown so far as the values are close to the value of 1. This is especially noted for F1-Score, Recall and precision. 

![image](https://user-images.githubusercontent.com/62455043/213055566-9512bdd1-40e9-4e93-b7d0-5e60f88ffa22.png)

Accuracy, Precision, Sensitivity, Specificity scores
The scores for accuracy, precision, sensitivity, and specificity are high. The main focus here is that the sensitivity score is high. This is beneficial for the result as FN is depicted as 0 meaning that the closer to 0 the FN value is the better. The reason for this is that it is ideal to reduce the possibility of a low sensitivity score in order to avoid incorrect classifications of Type II errors. 


### Business Value

Fraud prevention helps business by making it less easy for third parties to produce unauthorized transactions, however, for those cases where the prevention measures weren’t enough to halt the fraud from happening it’s transcendental to identify where irregularities were made to be able to take immediate actions seeking to revert the damage made. Fraud detection not only minimizes risk of economic loss from both company and customers, but also safeguards the integrity of the business brand value being this the bigger risk.

Given the need of ensuring safe online transactions it is required to develop techniques and mechanisms that will prevent businesses being affected by potential scams, protecting in this way, customers from unauthorized transactions but also the business reputation and image. For this reason, it is compulsory to use the latest technologies (Machine Learning) to prevent fraud through the implementation of the most accurate tools to detect and prevent any type of unauthorized transactions.

According to the Tech Guide of Fraud Risk Management, fraud prevention is the first measure to decrease incentives, restrict the opportunity and limit the ability for potential fraudsters. However, as fraud prevention may not necessarily stop all frauds from happening, it is necessary to have effective fraud detection measures in place. Both fraud prevention and fraud detection have a key role to play in order to minimize fraud risks and one could not succeed without the other. Fraud detection can not only help in stopping fraud to happen but also to find out offenses that already happened allowing business to revert those transactions.
The value proposal from this project comes through the implementation of the machine learning model developed to allow the company to identify the potential and committed unauthorized transactions. Granting the business with the technologic capacity to monitor the transactions being made and spot with a 99,4% of precision, fraudulent transactions.

Taking into consideration the historical data collected from the company for the analyzed period, by applying the suggested machine learning algorithm, the company could prevent a total of 8,163 from a total of 8,213 unauthorized transactions that were made during the recorded period. This means that from a total of 8,213 potential fraudulent transactions, only 49 transactions would actually happen. This improvement on detecting fraud would reduce the size of the impact considerably allowing the business to address individually each case with tailored specific solutions. Not only to safeguard the clients assets but also more important, the company integrity and brand image.

From a revenue point of view, by applying the proposed solution, given the average of 1.4 million EUR per fraudulent transaction, the company would save  11,983,014,621 EUR from stopping fraud.
The implementation of this business solution will free the budget to address the necessary solutions for the few fraudulent cases that were not spotted on time, while also protecting the business image.
