# Movie Recommender System

![streamlit_screen.png](attachment:streamlit_screen.png)

## Project Overview
___

Netflix is undeniably the biggest leader in the streaming world with a total global subscriber count of 232.5 million. One key behind Netflix's success is its innovative utilization of data science. By analyzing viewing habits, time spent, and content preferences, Netflix offers tailored recommendations of movies and TV shows for each user. These predictions lead to improved user experiences and increased engagement resulting in user retention, which is crucial for every business. 

The aim of this project is to develop a content-based recommender system that recommends similar movies to a user's input, by analyzing textual features such as title, genre, description, keywords and credits.

## Dataset Information
___

The dataset used in this project is Movies Daily Update Dataset from Kaggle. This dataset was made available through Akshay Pawar on Kaggle. This data is deemed credible as it operates under a public domain. The dataset contains metadata for more than 700,000 movies listed in the TMDB Dataset.

### Data Dictionary

| Variable           	| Definition                                      	| Key                                                 	|
|--------------------	|-------------------------------------------------	|-----------------------------------------------------	|
| id                 	| TMDB id                                         	|                                                     	|
| title              	| title of movie                                  	|                                                     	|
| genres             	| '-' separated category movie belongs to         	|                                                     	|
| original_language  	| langauge movie made in                          	|                                                     	|
| overview           	| short description of movie                      	|                                                     	|
| popularity         	| TMBD matric                                     	|                                                     	|
| production_company 	| '-' separated production company                	| 0=No, 1=Yes                                         	|
| budget             	| budget of movie                                 	| '1-2 Year', 'Less than 1 Year', 'More than 2 Years' 	|
| revenue            	| revenue genrated by movie                       	|                                                     	|
| runtime            	| duration of the movie                           	|                                                     	|
| status             	|                                                 	| 'Released', 'Planned', 'Other'                      	|
| tagline            	|                                                 	|                                                     	|
| vote_average       	| average of votes given by tmdb users            	|                                                     	|
| vote_count         	| vote counts                                     	|                                                     	|
| credits            	| '-' separated cast if movie                     	|                                                     	|
| keywords           	| '-' separated keywords that desciption of movie 	|                                                     	|
| poster_path        	| poster image                                    	|                                                     	|
| backdrop_path      	| background images                               	|                                                     	|
| recommendations    	| '-' separated recommended movie id              	|                                                     	|
| release_date       	| movie release date                              	|                                                     	|

## Approach
___

The project will involve several steps, including data preprocessing, exploratory data analysis, feature engineering, model selection, and evaluation. Techniques such as data cleaning, handling missing values, feature scaling and text vectorization will be employed to prepare the dataset for model training. For this project's machine learning model, we will use bag of words algorithm for vectorisation and cosine similarity for similarity analysis.

## Impact
___

A movie recommender system plays a crucial role in enhancing the user experience by providing personalized movie recommendations based on an individual's favorite movies. This system leverages advanced algorithms to analyze user preferences, viewing history, and movie attributes to suggest films that align with their tastes. By doing so, it not only saves users time and effort in searching for content but also introduces them to new movies they are likely to enjoy, thereby increasing engagement and satisfaction. Such a system is particularly beneficial in an era of vast and diverse content libraries, ensuring that users can easily discover films that match their interests and preferences.
