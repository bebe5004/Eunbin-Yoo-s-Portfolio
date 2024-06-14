# Heart Stroke Prediction

![heart_stroker](https://github.com/bebe5004/Eunbin-Yoo-s-Portfolio/assets/59913944/80c9b349-3acf-430c-8cb5-624b03af5482)


## Project Overview
---
This data science project aims to predict the probability of a patient experiencing a stroke based on various input parameters, including perfluorooctanoic acid (PFOA) levels, alongside a range of health, nutritional, and biochemical variables. The dataset contains relevant information for each patient, enabling the development of a predictive model.

## Dataset Information
---
The dataset used in this project contains data from four years of the NHANES health survey in the US. This survey measures a huge range of health, nutrition, and biochemical variables on a probability sample of the US population.

### Data Dictionary

| Variable 	| Definition                                                                                       	| Key                                                                                       	|
|----------	|--------------------------------------------------------------------------------------------------	|-------------------------------------------------------------------------------------------	|
| hadcvd   	| Logical. Did the person have diagnosed heart disease at or before the time of the survey?        	|                                                                                           	|
| pfoa4    	| Blood concentration of PFOA, split into four (approximately) equal categories at the quartiles.  	|                                                                                           	|
| DMDEDUC  	| Education                                                                                        	| 1=less than high school, 2=high school, 3= more than high school, 7=refused, 9=don't know 	|
| LBXGH    	| Blood concentration of glycosylated haemoglobin as a percentage of all haemoglobin.              	|                                                                                           	|
| RIDAGEEX 	| Age, in months, at the time of the survey                                                        	|                                                                                           	|
| RIDRETH1 	| Race / ethnicity                                                                                 	| 1=Mexican Hispanic, 2=Other Hispanic, 3=non-Hispanic White, 4=non-Hispanic Black, 5=other 	|
| RIAGENDR 	| Gender                                                                                           	| 1=male 2=female                                                                           	|
| BPXSAR   	| Systolic blood pressure                                                                          	|                                                                                           	|
| BPXDAR   	| Diastolic blood pressure                                                                         	|                                                                                           	|
| smoking  	| Never smoked, Current smoker, Former smoker                                                      	| 'Never', 'Current', 'Former'                                                              	|
| LBXTC    	| Blood cholesterol levels (mg/dL)                                                                 	|                                                                                           	|

## Context
___
According to the World Health Organization (WHO), stroke is the second leading cause of death worldwide, responsible for approximately 11% of all deaths. This project aims to utilize machine learning techniques to create a predictive model capable of identifying individuals at high risk of stroke by analyzing their demographic and health-related characteristics. By accurately identifying those at high risk early, healthcare providers can implement targeted preventive measures. This proactive approach has the potential to significantly reduce the incidence and severity of strokes, thereby improving health outcomes and saving lives.
