# Insurance Customer Segmentation Analysis

![market-segmentation.png](attachment:market-segmentation.png)

## Project Overview
___

Our client, an insurance company, has successfully provided health insurance to its customers and now seeks to understand which of these customers would be interested in their vehicle insurance offerings. Vehicle insurance, like health insurance, requires customers to pay an annual premium to the insurance provider, which then covers costs in case of vehicle accidents.

This project aims to conduct a segmentation analysis of health insurance customers to identify those who are also interested in vehicle insurance. Customer segmentation is crucial for businesses as it helps identify potential customers within a market by understanding the characteristics of existing customers. By analyzing these characteristics, the company can find individuals who share similarities with their loyal customers.

## Dataset Information
___
The dataset used in this project is downloaded from Health Insurance Cross Sell dataset uploaded on kaggle. This dataset involves information about demographics, vehicles and policy. The preprocessed and cleaned version of the dataset from Car Insurance Sales Project will be used.

### Data Dictionary

| Variable             	| Definition                                	| Key                                                 	|
|----------------------	|-------------------------------------------	|-----------------------------------------------------	|
| id                   	|                                           	|                                                     	|
| Gender               	| Gender of a customer                      	| 'Male', 'Female'                                    	|
| Age                  	| Age of a customer                         	|                                                     	|
| Driving_License      	| Does a customer have a driving license?   	| 0=No, 1=Yes                                         	|
| Region_Code          	|                                           	|                                                     	|
| Previously_Insured   	| Is a customer insured previously?         	| 0=No, 1=Yes                                         	|
| Vehicle_Age          	| Age of a customer's vehicle               	| '1-2 Year', 'Less than 1 Year', 'More than 2 Years' 	|
| Annual_Premium       	| Annual premium ammount                    	|                                                     	|
| Policy_Sales_Channel 	| Sales channel                             	|                                                     	|
| Vintage              	| Days customer associated for with company 	|                                                     	|
| Response             	|                                           	|                                                     	|

## Approach
___
The project will involve several steps, including data preprocessing, exploratory data analysis, feature engineering, clustering, model selection, and evaluation. Techniques such as data cleaning, handling missing values, feature scaling, and encoding categorical variables will be employed to prepare the dataset for model training. Various unsupervised machine learning algorithms, such as K-means Clustering and Principal Component Analysis (PCA) will be explored and evaluated to determine the most suitable model for effective segmentation.

## Impact
___

Through this project, we aim to provide actionable insights for the insurance company to strategically target health insurance customers who are likely to be interested in purchasing vehicle insurance. By segmenting customers based on their likelihood of buying vehicle insurance, the company can tailor communication strategies to meet specific segment needs, enhancing engagement and optimizing marketing efforts through effective channels.

This approach is expected to improve customer acquisition by focusing resources on individuals with higher conversion potential, thereby expanding the customer base and increasing revenue from vehicle insurance policies. Moreover, leveraging data-driven insights allows the company to refine product offerings and service delivery, boosting customer satisfaction and loyalty while driving sustainable business growth in a competitive insurance market.

## Business Questions
___

1. Can we identify distinct customer segments based on particular variables? 
2. What are the key characteristics of each customer segment?
3. Which segments are most valuable to the company in terms of revenue potential, loyalty, or growth opportunities?
4. Do different segments prefer different communication channels?
5. Are there differences in duration of a customer's association with the company across segments?
6. What are the age distribution? Are there the age-related trends and preferences?
