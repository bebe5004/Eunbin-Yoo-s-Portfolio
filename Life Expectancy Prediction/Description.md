# Life Expectancy Prediction

![life_expectancy](https://github.com/bebe5004/Eunbin-Yoo-s-Portfolio/assets/59913944/5a1018dc-9b52-4d20-971b-3159bf8f4888)


## Project Overview

There is little doubt that life expectancy increased in the past decade. Why is life expectancy longer now? What influences our lifespan? This project aims to explore the WHO Life Expectancy dataset to investigate trends and disparities in global life expectancy through analysis and visualizations using python libraries, matplotlib, seaborn and plotly. The project also aims to build a regression model to predict life expectancy of a country, by analyzing various demographic, socio-economic and health variables.

## Dataset Information

The dataset used in this project is Life Expectancy (WHO) dataset from Kaggle. This dataset contains observations on factors related to life expectancy from a period of 2000 to 2015 for all the countries, such as immunization factors, mortality factors, economic factors, social factors and other health related factors as well.

### Data Dictionary

| Variable                        	| Definition                                                                                             	| Key                       	|
|---------------------------------	|--------------------------------------------------------------------------------------------------------	|---------------------------	|
| id                              	| TMDB id                                                                                                	|                           	|
| Country                         	|                                                                                                        	|                           	|
| Year                            	|                                                                                                        	|                           	|
| Status                          	| Developed or developing country                                                                        	| 'Developed', 'Developing' 	|
| Life expectancy                 	| Life expectancy in age                                                                                 	|                           	|
| Adult Mortality                 	| Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population) 	|                           	|
| Infant deaths                   	| Number of Infant Deaths per 1000 population                                                            	|                           	|
| Alcohol                         	| Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)                             	|                           	|
| percentage expenditure          	| Expenditure on health as a percentage of Gross Domestic Product per capita(%)                          	|                           	|
| Hepatitis B                     	| Hepatitis B (HepB) immunization coverage among 1-year-olds (%)                                         	|                           	|
| Measles                         	| Measles - number of reported cases per 1000 population                                                 	|                           	|
| BMI                             	| Average Body Mass Index of entire population                                                           	|                           	|
| under-five deaths               	| Number of under-five deaths per 1000 population                                                        	|                           	|
| Polio                           	| Polio (Pol3) immunization coverage among 1-year-olds (%)                                               	|                           	|
| Total expenditure               	| General government expenditure on health as a percentage of total government expenditure (%)           	|                           	|
| Diphtheria                      	| Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)             	|                           	|
| HIV/AIDS                        	| Deaths per 1 000 live births HIV/AIDS (0-4 years)                                                      	|                           	|
| GDP                             	| Gross Domestic Product per capita (in USD)                                                             	|                           	|
| thinness 1-19 years old         	| Prevalence of thinness among children and adolescents for Age 10 to 19 (% )                            	|                           	|
| thinness 5-9 years old          	| Prevalence of thinness among children for Age 5 to 9(%)                                                	|                           	|
| Income composition of resources 	| Human Development Index in terms of income composition of resources (index ranging from 0 to 1)        	|                           	|
| Schooling                       	| Number of years of Schooling(years)                                                                    	|                           	|

## Approach

The project will involve several steps, including data preprocessing, exploratory data analysis, feature engineering, model selection, and evaluation. Techniques such as data cleaning, handling missing values, feature scaling, and encoding categorical variables will be employed to prepare the dataset for model training. Various regression algorithms, such as linear regression, random forests will be explored and evaluated to determine the most suitable model for accurate life expectancy prediction.

## Impact

The development of a machine learning model to predict life expectancy can significantly benefit countries with lower life expectancy by identifying key predictive factors from a diverse dataset. This enables nations to pinpoint critical areas needing improvement, such as healthcare access, education levels, or nutrition. By providing targeted insights, the model helps governments formulate informed strategies and policies, allocate resources effectively, and implement relevant programs. This focused approach ensures efforts are directed towards the most impactful areas, ultimately leading to improved health outcomes, reduced mortality rates, and a healthier, more productive society.

## Questions

These questions will be answered throughout the project:
1. What are the predicting variables actually affecting the life expectancy?
2. Should a country having a lower life expectancy value(<65) increase its healthcare expenditure in order to improve its average lifespan?
3. How does Infant and Adult mortality rates affect life expectancy?
4. What is the impact of schooling on the lifespan of humans?
5. Does Life Expectancy have positive or negative relationship with drinking alcohol?
