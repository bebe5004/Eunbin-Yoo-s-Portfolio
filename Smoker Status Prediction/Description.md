# Binary Prediction of Smoker Status using Bio-Signals

![smoke](https://github.com/bebe5004/Eunbin-Yoo-s-Portfolio/assets/59913944/a417c0dc-8c40-4ace-ba6d-e9e8d399d671)

## Project Overview

Smoking negatively affects our physical health in many ways. It harms nearly every organ in the body. How much can an individual's health conditions tell about their smoking status? This data science project aims to create a binary classification model to predict the smoker status of an individual, by analyzing a wide range of demographic and health variables such as age, body size, eyesight, hearing, blood pressure, fasting blood sugar and cholestrol. The project also aims to find out the set of variables that has the most impact on the smoker status.

## Dataset Information

The dataset used in this project consists of a wide range of bio-signals that can potentially be influenced by smoking status, such as height, weight, waist, eyesight, hearing, fasting blood sugar and systolic blood pressure. Here's the information about the first 10 variables of the dataset.

### Data Dictionary

| Variable            	| Definition                             	| Key 	|
|---------------------	|----------------------------------------	|-----	|
| id                  	|                                        	|     	|
| age                 	| 5-years gap                            	|     	|
| height(cm)          	|                                        	|     	|
| weight(kg)          	|                                        	|     	|
| waist(cm)           	| Waist circumference length             	|     	|
| eyesight(left)      	|                                        	|     	|
| eyesight(right)     	|                                        	|     	|
| hearing(left)       	|                                        	|     	|
| hearing(right)      	|                                        	|     	|
| systolic            	| Blood pressure                         	|     	|
| relaxation          	| Blood pressure                         	|     	|
| fasting blood sugar 	|                                        	|     	|
| Cholesterol         	| Total                                  	|     	|
| triglyceride        	|                                        	|     	|
| HDL                 	| Cholesterol type                       	|     	|
| LDL                 	| Cholesterol type                       	|     	|
| hemoglobin          	|                                        	|     	|
| Urine protein       	|                                        	|     	|
| serum creatinine    	|                                        	|     	|
| AST                 	| Glutamic oxaloacetic transaminase type 	|     	|
| ALT                 	| Glutamic oxaloacetic transaminase type 	|     	|
| Gtp                 	| γ-GTP                                  	|     	|
| dental caries       	|                                        	|     	|
| smoking             	|                                        	|     	|

## Approach

The project will involve several steps, including data preprocessing, exploratory data analysis, feature engineering, model selection, and evaluation. Techniques such as data cleaning, handling missing values and feature scaling will be employed to prepare the dataset for model training. Various regression algorithms, such as logistic linear regression, random forests classifier will be explored and evaluated to determine the most suitable model for accurate classification of smoker status.

## Context

Smoking has been proven to negatively affect health in a multitude of ways. Smoking has been found to harm nearly every organ of the body, cause many diseases, as well as reducing the life expectancy of smokers in general. As of 2018, smoking has been considered the leading cause of preventable morbidity and mortality in the world, continuing to plague the world's overall health. According to a World Health Organization report, the number of deaths caused by smoking will reach 10 million by 2030. Evidence-based treatment for assistance in smoking cessation had been proposed and promoted. Providing a prediction model might be a favorable way to understand the chance of quitting smoking for each individual smoker.
