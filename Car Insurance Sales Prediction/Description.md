# Car Insurance Sales Prediction

![car](https://github.com/bebe5004/Eunbin-Yoo-s-Portfolio/assets/59913944/677aedae-8e84-42fb-8f76-f28432c6ec38)

## Project Overview

Our client, an insurance company, seeks to expand its offerings by introducing vehicle insurance to its existing health insurance customers. To achieve this, we aim to build a predictive model to determine whether policyholders from the past year will be interested in purchasing vehicle insurance.

The objective of this project is to leverage machine learning techniques to predict customer interest in vehicle insurance. By analyzing a dataset that includes demographics (gender, age, region code type), vehicle details (vehicle age, damage history), and policy specifics (premium amount, sourcing channel), we aim to develop a model that accurately forecasts which customers are likely to purchase vehicle insurance.

## Dataset Information

The dataset used in this project is downloaded from Health Insurance Cross Sell dataset uploaded on kaggle. This dataset involves information about demographics, vehicles and policy. 

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

The project will involve several steps, including data preprocessing, exploratory data analysis, feature engineering, model selection, and evaluation. Techniques such as data cleaning, handling missing values, feature scaling, and encoding categorical variables will be employed to prepare the dataset for model training. Various regression algorithms, such as logistic linear regression, random forests classifier will be explored and evaluated to determine the most suitable model for accurate prediction.

## Impact

This project is invaluable for the company as it enables the development of a targeted communication strategy, allowing for precise outreach to customers most likely to be interested in vehicle insurance. By leveraging predictive insights, the company can craft personalized marketing messages, enhancing customer engagement and conversion rates. Additionally, understanding customer preferences and behaviors aids in optimizing the overall business model, ensuring efficient resource allocation and streamlined operations. This data-driven approach not only reduces marketing expenditures but also maximizes revenue potential by focusing efforts on high-probability customers. Consequently, the company can enhance its profitability, expand its customer base, and strengthen its market position, leading to sustained business growth and competitive advantage.
