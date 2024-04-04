# Exploratory Data Analysis on Netflix

# 1. Introduction

Thanks to the rise of streaming services, the way people consume entertainment has changed dramatically in recent years. Netflix is undeniably the biggest leader in the streaming world with a total global subscriber count of 232.5 million. In this project, I will explore the Netflix dataset through visualizations and graphs using python libraries, matplotlib and seaborn. I used Netflix Movies and TV Shows dataset from Kaggle. The dataset consists of listings of all TV Shows and Movies available on Netflix as of mid-2021.

## Import Libraries


```python
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import re
%matplotlib inline
```


```python
from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")
```

    C:\Users\eunbi\AppData\Local\Temp\ipykernel_19832\2578555594.py:2: DeprecationWarning: `set_matplotlib_formats` is deprecated since IPython 7.23, directly use `matplotlib_inline.backend_inline.set_matplotlib_formats()`
      set_matplotlib_formats("retina")
    

# Table of Contents

1. Data Preparation and Cleaning
2. Exploratory Analysis and Visualization
3. Conclusions

# 2. Data Preparation and Cleaning

## Loading the Dataset
I'll load the CSV file using Pandas Library. I named the imported dataset with netflix.


```python
netflix = pd.read_csv(r"C:\Users\eunbi\Desktop\DS\Projects\Netflix titles EDA\netflix_titles.csv")
netflix.shape
```




    (8807, 12)




```python
netflix.nunique()
```




    show_id         8807
    type               2
    title           8807
    director        4528
    cast            7692
    country          748
    date_added      1767
    release_year      74
    rating            17
    duration         220
    listed_in        514
    description     8775
    dtype: int64



Let's check the first 5 data.


```python
netflix.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>Kirsten Johnson</td>
      <td>NaN</td>
      <td>United States</td>
      <td>September 25, 2021</td>
      <td>2020</td>
      <td>PG-13</td>
      <td>90 min</td>
      <td>Documentaries</td>
      <td>As her father nears the end of his life, filmm...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s2</td>
      <td>TV Show</td>
      <td>Blood &amp; Water</td>
      <td>NaN</td>
      <td>Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban...</td>
      <td>South Africa</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
      <td>International TV Shows, TV Dramas, TV Mysteries</td>
      <td>After crossing paths at a party, a Cape Town t...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s3</td>
      <td>TV Show</td>
      <td>Ganglands</td>
      <td>Julien Leclercq</td>
      <td>Sami Bouajila, Tracy Gotoas, Samuel Jouy, Nabi...</td>
      <td>NaN</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>1 Season</td>
      <td>Crime TV Shows, International TV Shows, TV Act...</td>
      <td>To protect his family from a powerful drug lor...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s4</td>
      <td>TV Show</td>
      <td>Jailbirds New Orleans</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>1 Season</td>
      <td>Docuseries, Reality TV</td>
      <td>Feuds, flirtations and toilet talk go down amo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s5</td>
      <td>TV Show</td>
      <td>Kota Factory</td>
      <td>NaN</td>
      <td>Mayur More, Jitendra Kumar, Ranjan Raj, Alam K...</td>
      <td>India</td>
      <td>September 24, 2021</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
      <td>International TV Shows, Romantic TV Shows, TV ...</td>
      <td>In a city of coaching centers known to train I...</td>
    </tr>
  </tbody>
</table>
</div>



The dataset contains 8807 different movie/TVshow data: 6126 Movie data and 2664 TV Show data. The columns consist of 12 typical movie/TVshows descriptions, such as type, title, director, cast, country, date added and release year. We can see that there are NaN values in some columns.

## Handling missing data
Let's have a look at the missing data.


```python
netflix.isnull().sum().sum()
```




    4307




```python
netflix.isnull().sum()
```




    show_id            0
    type               0
    title              0
    director        2634
    cast             825
    country          831
    date_added        10
    release_year       0
    rating             4
    duration           3
    listed_in          0
    description        0
    dtype: int64




```python
import missingno as msno

msno.matrix(netflix)
plt.title("Visualization of the Nullity of The Data", fontsize=30)
```




    Text(0.5, 1.0, 'Visualization of the Nullity of The Data')




    
![png](output_16_1.png)
    


There are a total of 4307 missing values in the data. The visualization above shows that there are a lot of missing values in "director", "cast" and "country" columns. "data_added", "rating" and "duration" also have a few missing values.

The easiest way to get rid of them would be to delete the rows with the missing data. However, this is not the best option for this EDA. Since the number of missing values is large, dropping them would result in a significant loss of information. 

For the columns containing the majority of null values, such as "director", "cast", "country" and "rating", I chose to fill each missing value with 'unavailable' using fillna() function. "date_added", "duration" and "rating" can be dropped from the dataset because they contain an insignificant portion of the data.


```python
netflix["director"] = netflix["director"].fillna("Unavailable")
netflix["cast"] = netflix["cast"].fillna("Unavailable")
netflix["country"] = netflix["country"].fillna("Unavailable")
netflix.dropna(subset=["date_added", "duration", "rating"], inplace=True)
```


```python
netflix.isnull().sum()
```




    show_id         0
    type            0
    title           0
    director        0
    cast            0
    country         0
    date_added      0
    release_year    0
    rating          0
    duration        0
    listed_in       0
    description     0
    dtype: int64




```python
netflix.shape
```




    (8790, 12)



Finally, we can see that there are no more missing values in the data. The data size is reduced to 8790.

## Dates data
The datatype of "date_added" is object type. Transforming this variable to datetime data will be helpful. I will extract year and month from "date added" and add new columns "year_added" and "month_added".


```python
netflix["date_added"] = pd.to_datetime(netflix["date_added"])
netflix["year_added"] = netflix["date_added"].dt.year
netflix["month_added"] = netflix["date_added"].dt.month
netflix.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>show_id</th>
      <th>type</th>
      <th>title</th>
      <th>director</th>
      <th>cast</th>
      <th>country</th>
      <th>date_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
      <th>listed_in</th>
      <th>description</th>
      <th>year_added</th>
      <th>month_added</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s1</td>
      <td>Movie</td>
      <td>Dick Johnson Is Dead</td>
      <td>Kirsten Johnson</td>
      <td>Unavailable</td>
      <td>United States</td>
      <td>2021-09-25</td>
      <td>2020</td>
      <td>PG-13</td>
      <td>90 min</td>
      <td>Documentaries</td>
      <td>As her father nears the end of his life, filmm...</td>
      <td>2021</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



## Unnesting columns

Some columns, such as "cast" and "country", contain a list of string values. Unnesting these columns will be possible. First, I want to create separate lines for each cast member in a movie.


```python
# unnesting "cast" columns
constraint=netflix["cast"].apply(lambda x: str(x).split(', ')).tolist()
actors_un = pd.DataFrame(constraint,index=netflix["title"])
actors_un = actors_un.stack()
actors_un = pd.DataFrame(actors_un.reset_index())
actors_un.rename(columns={0:'Actors'},inplace=True)
actors_un.drop(['level_1'],axis=1,inplace=True)
actors_un.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>Actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dick Johnson Is Dead</td>
      <td>Unavailable</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blood &amp; Water</td>
      <td>Ama Qamata</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blood &amp; Water</td>
      <td>Khosi Ngema</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Blood &amp; Water</td>
      <td>Gail Mabalane</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Blood &amp; Water</td>
      <td>Thabang Molaba</td>
    </tr>
  </tbody>
</table>
</div>



A lot of contents are listed in multiple genres. I want to create separate lines for each genre as well.


```python
# unnesting "listed_in" columns
constraint2 = netflix["listed_in"].apply(lambda x: str(x).split(', ')).tolist()
genres_un = pd.DataFrame(constraint2,index=netflix["title"])
genres_un = genres_un.stack()
genres_un = pd.DataFrame(genres_un.reset_index())
genres_un.rename(columns={0:"Genre"},inplace=True)
genres_un.drop(["level_1"],axis=1,inplace=True)
genres_un.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>Genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dick Johnson Is Dead</td>
      <td>Documentaries</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blood &amp; Water</td>
      <td>International TV Shows</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blood &amp; Water</td>
      <td>TV Dramas</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Blood &amp; Water</td>
      <td>TV Mysteries</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ganglands</td>
      <td>Crime TV Shows</td>
    </tr>
  </tbody>
</table>
</div>



Lastly, I'm going to create separate lines for each country in a movie.


```python
# unnesting "country" columns
constraint3 = netflix["country"].apply(lambda x: str(x).split(', ')).tolist()
country_un = pd.DataFrame(constraint3, index=netflix["title"])
country_un = country_un.stack()
country_un = pd.DataFrame(country_un.reset_index())
country_un.rename(columns={0:"Country"}, inplace=True)
country_un.drop(["level_1"], axis=1, inplace=True)
country_un.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dick Johnson Is Dead</td>
      <td>United States</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blood &amp; Water</td>
      <td>South Africa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ganglands</td>
      <td>Unavailable</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jailbirds New Orleans</td>
      <td>Unavailable</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kota Factory</td>
      <td>India</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merge unnested actors data with unnested genres data
netflix_un = actors_un.merge(genres_un, on=["title"], how="inner")
# merge the above data with unnested  countries data
netflix_un = netflix_un.merge(country_un, on=["title"], how="inner")
# merge the anove data with the original data 
cols = ["show_id", "director", "type", "title", "date_added", "year_added", "month_added", "release_year", "rating", "duration"]
netflix_new = netflix_un.merge(netflix[cols], on=["title"], how="left")
netflix_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>Actors</th>
      <th>Genre</th>
      <th>Country</th>
      <th>show_id</th>
      <th>director</th>
      <th>type</th>
      <th>date_added</th>
      <th>year_added</th>
      <th>month_added</th>
      <th>release_year</th>
      <th>rating</th>
      <th>duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dick Johnson Is Dead</td>
      <td>Unavailable</td>
      <td>Documentaries</td>
      <td>United States</td>
      <td>s1</td>
      <td>Kirsten Johnson</td>
      <td>Movie</td>
      <td>2021-09-25</td>
      <td>2021</td>
      <td>9</td>
      <td>2020</td>
      <td>PG-13</td>
      <td>90 min</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Blood &amp; Water</td>
      <td>Ama Qamata</td>
      <td>International TV Shows</td>
      <td>South Africa</td>
      <td>s2</td>
      <td>Unavailable</td>
      <td>TV Show</td>
      <td>2021-09-24</td>
      <td>2021</td>
      <td>9</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Blood &amp; Water</td>
      <td>Ama Qamata</td>
      <td>TV Dramas</td>
      <td>South Africa</td>
      <td>s2</td>
      <td>Unavailable</td>
      <td>TV Show</td>
      <td>2021-09-24</td>
      <td>2021</td>
      <td>9</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Blood &amp; Water</td>
      <td>Ama Qamata</td>
      <td>TV Mysteries</td>
      <td>South Africa</td>
      <td>s2</td>
      <td>Unavailable</td>
      <td>TV Show</td>
      <td>2021-09-24</td>
      <td>2021</td>
      <td>9</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Blood &amp; Water</td>
      <td>Khosi Ngema</td>
      <td>International TV Shows</td>
      <td>South Africa</td>
      <td>s2</td>
      <td>Unavailable</td>
      <td>TV Show</td>
      <td>2021-09-24</td>
      <td>2021</td>
      <td>9</td>
      <td>2021</td>
      <td>TV-MA</td>
      <td>2 Seasons</td>
    </tr>
  </tbody>
</table>
</div>



Now, it's finally time to explore the data and create visualizations.

# 3. Exploratory Analysis and Visualization

## 3.1 How many Movies and TV Shows on Netflix?

The dataset consists of 6126 movie data and 2664 TV show data. We already know that there are more movie data than TV show data, but I want to compare them with ratio. Pie chart would be a good choice for the proportions of categorical data.


```python
# Calculate the ratio of Movies and TV Shows
label = ["Movie", "TV Shows"]
mtratio = netflix_new["type"].value_counts(normalize=True)
# Draw a pie chart indicating the percentage of Moves and TV Shows
plt.rcParams["figure.figsize"] = (15,9)
plt.pie(mtratio, labels=label, colors=["red", "black"], counterclock=False, 
        explode=(0.025,0.025), textprops={'fontsize': 18},
        autopct='%.1f%%')
plt.title("Distribution of Movies and TV Shows on Netflix", fontsize=24)
plt.show()
```


    
![png](output_34_0.png)
    


There are more than 6000 movies and almost 3,000 TV shows on Netflix, with movies being the majority. There are far more movie titles (69.7%) than TV shows titles (30.3%).

## 3.2 How has the amount of content on Netflix changed over time?
Now, I will explore the amount of content Netflix has added over the previous years. I will extract year and month from "date added" and add new columns "year_added" and "month_added". 


```python
# Count the amount of content by "year_added" and "type"
year_type = netflix_new.groupby(["year_added", "type"])["title"].count().unstack()
year_type = year_type.reset_index()
year_type = year_type.fillna(0)
year_type["Total"] = year_type["Movie"] + year_type["TV Show"]
```


```python
# Draw a line plot with "year_added" on the x-axis
yeartype_plt = year_type[:-1].plot.line(x="year_added", fontsize=14,
                                          figsize=(15,8), 
                                          title="Total content added on Netflix across the years(~2020)")
yeartype_plt.title.set_size(24)
yeartype_plt.legend(title="", fontsize=14)
yeartype_plt.set_xlabel("Year", fontsize=18)
yeartype_plt.set_ylabel("Releases", fontsize=18)
matplotlib.style.use("seaborn-white")
```

    C:\Users\eunbi\AppData\Local\Temp\ipykernel_19832\3074331866.py:9: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      matplotlib.style.use("seaborn-white")
    


    
![png](output_38_1.png)
    


Based on the line plot above, we can say that Netflix started growing its business as a streaming platform from 2013. Since 2015, the amount of content added has been rapidly increasing. Besides, it clearly appears that Netflix has increasingly focused on TV shows rather than movies in recent years. While the number of TV show releases kept increasing, the amount of increase in the number of movie releases has reduced since 2017 and the number even dropped in 2020.


```python
year_month = pd.pivot_table(netflix_new, index="month_added", columns="year_added", values="title", aggfunc="count")
year_month = year_month.fillna(0)
year_month.style.background_gradient("Reds")
```




<style type="text/css">
#T_c8b0a_row0_col0, #T_c8b0a_row0_col12, #T_c8b0a_row4_col1, #T_c8b0a_row6_col13, #T_c8b0a_row8_col3, #T_c8b0a_row9_col9, #T_c8b0a_row9_col10, #T_c8b0a_row10_col2, #T_c8b0a_row10_col11, #T_c8b0a_row11_col4, #T_c8b0a_row11_col5, #T_c8b0a_row11_col6, #T_c8b0a_row11_col7, #T_c8b0a_row11_col8 {
  background-color: #67000d;
  color: #f1f1f1;
}
#T_c8b0a_row0_col1, #T_c8b0a_row0_col2, #T_c8b0a_row0_col3, #T_c8b0a_row0_col4, #T_c8b0a_row0_col5, #T_c8b0a_row0_col7, #T_c8b0a_row1_col1, #T_c8b0a_row1_col2, #T_c8b0a_row1_col3, #T_c8b0a_row1_col5, #T_c8b0a_row2_col0, #T_c8b0a_row2_col1, #T_c8b0a_row2_col2, #T_c8b0a_row2_col3, #T_c8b0a_row2_col4, #T_c8b0a_row2_col6, #T_c8b0a_row3_col0, #T_c8b0a_row3_col1, #T_c8b0a_row3_col2, #T_c8b0a_row3_col3, #T_c8b0a_row3_col4, #T_c8b0a_row3_col5, #T_c8b0a_row4_col0, #T_c8b0a_row4_col2, #T_c8b0a_row4_col4, #T_c8b0a_row4_col5, #T_c8b0a_row4_col6, #T_c8b0a_row4_col11, #T_c8b0a_row5_col0, #T_c8b0a_row5_col1, #T_c8b0a_row5_col2, #T_c8b0a_row5_col3, #T_c8b0a_row5_col4, #T_c8b0a_row5_col5, #T_c8b0a_row5_col8, #T_c8b0a_row5_col10, #T_c8b0a_row6_col0, #T_c8b0a_row6_col1, #T_c8b0a_row6_col2, #T_c8b0a_row6_col3, #T_c8b0a_row6_col4, #T_c8b0a_row6_col5, #T_c8b0a_row7_col0, #T_c8b0a_row7_col1, #T_c8b0a_row7_col2, #T_c8b0a_row7_col3, #T_c8b0a_row7_col4, #T_c8b0a_row7_col12, #T_c8b0a_row8_col0, #T_c8b0a_row8_col1, #T_c8b0a_row8_col2, #T_c8b0a_row8_col4, #T_c8b0a_row9_col0, #T_c8b0a_row9_col1, #T_c8b0a_row9_col2, #T_c8b0a_row9_col4, #T_c8b0a_row9_col13, #T_c8b0a_row10_col0, #T_c8b0a_row10_col3, #T_c8b0a_row10_col9, #T_c8b0a_row10_col13, #T_c8b0a_row11_col0, #T_c8b0a_row11_col1, #T_c8b0a_row11_col2, #T_c8b0a_row11_col3, #T_c8b0a_row11_col13 {
  background-color: #fff5f0;
  color: #000000;
}
#T_c8b0a_row0_col6 {
  background-color: #77040f;
  color: #f1f1f1;
}
#T_c8b0a_row0_col8 {
  background-color: #fc8666;
  color: #f1f1f1;
}
#T_c8b0a_row0_col9 {
  background-color: #ffebe2;
  color: #000000;
}
#T_c8b0a_row0_col10, #T_c8b0a_row8_col12 {
  background-color: #fc9272;
  color: #000000;
}
#T_c8b0a_row0_col11 {
  background-color: #fcb89e;
  color: #000000;
}
#T_c8b0a_row0_col13 {
  background-color: #fb7a5a;
  color: #f1f1f1;
}
#T_c8b0a_row1_col0 {
  background-color: #ffece3;
  color: #000000;
}
#T_c8b0a_row1_col4 {
  background-color: #fee3d6;
  color: #000000;
}
#T_c8b0a_row1_col6, #T_c8b0a_row4_col3 {
  background-color: #f85d42;
  color: #f1f1f1;
}
#T_c8b0a_row1_col7 {
  background-color: #fee0d2;
  color: #000000;
}
#T_c8b0a_row1_col8, #T_c8b0a_row1_col12, #T_c8b0a_row3_col9, #T_c8b0a_row6_col9 {
  background-color: #fff4ef;
  color: #000000;
}
#T_c8b0a_row1_col9 {
  background-color: #fdcdb9;
  color: #000000;
}
#T_c8b0a_row1_col10, #T_c8b0a_row3_col7 {
  background-color: #fee3d7;
  color: #000000;
}
#T_c8b0a_row1_col11 {
  background-color: #fee5d9;
  color: #000000;
}
#T_c8b0a_row1_col13 {
  background-color: #fc8a6a;
  color: #f1f1f1;
}
#T_c8b0a_row2_col5, #T_c8b0a_row7_col5 {
  background-color: #fb7252;
  color: #f1f1f1;
}
#T_c8b0a_row2_col7 {
  background-color: #fedaca;
  color: #000000;
}
#T_c8b0a_row2_col8 {
  background-color: #fff0e8;
  color: #000000;
}
#T_c8b0a_row2_col9 {
  background-color: #f14331;
  color: #f1f1f1;
}
#T_c8b0a_row2_col10 {
  background-color: #a60f15;
  color: #f1f1f1;
}
#T_c8b0a_row2_col11 {
  background-color: #fc8060;
  color: #f1f1f1;
}
#T_c8b0a_row2_col12 {
  background-color: #fff3ed;
  color: #000000;
}
#T_c8b0a_row2_col13 {
  background-color: #fc9373;
  color: #000000;
}
#T_c8b0a_row3_col6 {
  background-color: #fcc3ab;
  color: #000000;
}
#T_c8b0a_row3_col8, #T_c8b0a_row7_col8 {
  background-color: #fee7db;
  color: #000000;
}
#T_c8b0a_row3_col10 {
  background-color: #fc9c7d;
  color: #000000;
}
#T_c8b0a_row3_col11, #T_c8b0a_row5_col11 {
  background-color: #fdc9b3;
  color: #000000;
}
#T_c8b0a_row3_col12 {
  background-color: #fc8262;
  color: #f1f1f1;
}
#T_c8b0a_row3_col13 {
  background-color: #ec382b;
  color: #f1f1f1;
}
#T_c8b0a_row4_col7, #T_c8b0a_row4_col12 {
  background-color: #fdc5ae;
  color: #000000;
}
#T_c8b0a_row4_col8, #T_c8b0a_row8_col11 {
  background-color: #fff1ea;
  color: #000000;
}
#T_c8b0a_row4_col9 {
  background-color: #fdd5c4;
  color: #000000;
}
#T_c8b0a_row4_col10 {
  background-color: #fee2d5;
  color: #000000;
}
#T_c8b0a_row4_col13, #T_c8b0a_row9_col11 {
  background-color: #fb7757;
  color: #f1f1f1;
}
#T_c8b0a_row5_col6, #T_c8b0a_row9_col6 {
  background-color: #fdcab5;
  color: #000000;
}
#T_c8b0a_row5_col7 {
  background-color: #fcab8f;
  color: #000000;
}
#T_c8b0a_row5_col9 {
  background-color: #fb6d4d;
  color: #f1f1f1;
}
#T_c8b0a_row5_col12 {
  background-color: #fc8969;
  color: #f1f1f1;
}
#T_c8b0a_row5_col13 {
  background-color: #dd2a25;
  color: #f1f1f1;
}
#T_c8b0a_row6_col6 {
  background-color: #ffeee7;
  color: #000000;
}
#T_c8b0a_row6_col7 {
  background-color: #fcb499;
  color: #000000;
}
#T_c8b0a_row6_col8 {
  background-color: #fcb296;
  color: #000000;
}
#T_c8b0a_row6_col10 {
  background-color: #b11218;
  color: #f1f1f1;
}
#T_c8b0a_row6_col11 {
  background-color: #fee5d8;
  color: #000000;
}
#T_c8b0a_row6_col12 {
  background-color: #fee1d3;
  color: #000000;
}
#T_c8b0a_row7_col6 {
  background-color: #fdccb8;
  color: #000000;
}
#T_c8b0a_row7_col7 {
  background-color: #fedfd0;
  color: #000000;
}
#T_c8b0a_row7_col9 {
  background-color: #cb181d;
  color: #f1f1f1;
}
#T_c8b0a_row7_col10 {
  background-color: #aa1016;
  color: #f1f1f1;
}
#T_c8b0a_row7_col11 {
  background-color: #fee8dd;
  color: #000000;
}
#T_c8b0a_row7_col13 {
  background-color: #eb372a;
  color: #f1f1f1;
}
#T_c8b0a_row8_col5 {
  background-color: #c2161b;
  color: #f1f1f1;
}
#T_c8b0a_row8_col6 {
  background-color: #fff4ee;
  color: #000000;
}
#T_c8b0a_row8_col7 {
  background-color: #fc8d6d;
  color: #f1f1f1;
}
#T_c8b0a_row8_col8 {
  background-color: #fcae92;
  color: #000000;
}
#T_c8b0a_row8_col9 {
  background-color: #ab1016;
  color: #f1f1f1;
}
#T_c8b0a_row8_col10 {
  background-color: #fb6c4c;
  color: #f1f1f1;
}
#T_c8b0a_row8_col13 {
  background-color: #e32f27;
  color: #f1f1f1;
}
#T_c8b0a_row9_col3 {
  background-color: #d52221;
  color: #f1f1f1;
}
#T_c8b0a_row9_col5 {
  background-color: #fc8b6b;
  color: #f1f1f1;
}
#T_c8b0a_row9_col7 {
  background-color: #bc141a;
  color: #f1f1f1;
}
#T_c8b0a_row9_col8 {
  background-color: #f34935;
  color: #f1f1f1;
}
#T_c8b0a_row9_col12 {
  background-color: #fc8565;
  color: #f1f1f1;
}
#T_c8b0a_row10_col1 {
  background-color: #fcbba1;
  color: #000000;
}
#T_c8b0a_row10_col4 {
  background-color: #9d0d14;
  color: #f1f1f1;
}
#T_c8b0a_row10_col5 {
  background-color: #fedbcc;
  color: #000000;
}
#T_c8b0a_row10_col6 {
  background-color: #fb694a;
  color: #f1f1f1;
}
#T_c8b0a_row10_col7 {
  background-color: #fee4d8;
  color: #000000;
}
#T_c8b0a_row10_col8 {
  background-color: #fcb99f;
  color: #000000;
}
#T_c8b0a_row10_col10 {
  background-color: #d21f20;
  color: #f1f1f1;
}
#T_c8b0a_row10_col12 {
  background-color: #fcc1a8;
  color: #000000;
}
#T_c8b0a_row11_col9 {
  background-color: #e12d26;
  color: #f1f1f1;
}
#T_c8b0a_row11_col10 {
  background-color: #9f0e14;
  color: #f1f1f1;
}
#T_c8b0a_row11_col11 {
  background-color: #ce1a1e;
  color: #f1f1f1;
}
#T_c8b0a_row11_col12 {
  background-color: #fca98c;
  color: #000000;
}
</style>
<table id="T_c8b0a">
  <thead>
    <tr>
      <th class="index_name level0" >year_added</th>
      <th id="T_c8b0a_level0_col0" class="col_heading level0 col0" >2008</th>
      <th id="T_c8b0a_level0_col1" class="col_heading level0 col1" >2009</th>
      <th id="T_c8b0a_level0_col2" class="col_heading level0 col2" >2010</th>
      <th id="T_c8b0a_level0_col3" class="col_heading level0 col3" >2011</th>
      <th id="T_c8b0a_level0_col4" class="col_heading level0 col4" >2012</th>
      <th id="T_c8b0a_level0_col5" class="col_heading level0 col5" >2013</th>
      <th id="T_c8b0a_level0_col6" class="col_heading level0 col6" >2014</th>
      <th id="T_c8b0a_level0_col7" class="col_heading level0 col7" >2015</th>
      <th id="T_c8b0a_level0_col8" class="col_heading level0 col8" >2016</th>
      <th id="T_c8b0a_level0_col9" class="col_heading level0 col9" >2017</th>
      <th id="T_c8b0a_level0_col10" class="col_heading level0 col10" >2018</th>
      <th id="T_c8b0a_level0_col11" class="col_heading level0 col11" >2019</th>
      <th id="T_c8b0a_level0_col12" class="col_heading level0 col12" >2020</th>
      <th id="T_c8b0a_level0_col13" class="col_heading level0 col13" >2021</th>
    </tr>
    <tr>
      <th class="index_name level0" >month_added</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
      <th class="blank col8" >&nbsp;</th>
      <th class="blank col9" >&nbsp;</th>
      <th class="blank col10" >&nbsp;</th>
      <th class="blank col11" >&nbsp;</th>
      <th class="blank col12" >&nbsp;</th>
      <th class="blank col13" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c8b0a_level0_row0" class="row_heading level0 row0" >1</th>
      <td id="T_c8b0a_row0_col0" class="data row0 col0" >18.000000</td>
      <td id="T_c8b0a_row0_col1" class="data row0 col1" >0.000000</td>
      <td id="T_c8b0a_row0_col2" class="data row0 col2" >0.000000</td>
      <td id="T_c8b0a_row0_col3" class="data row0 col3" >0.000000</td>
      <td id="T_c8b0a_row0_col4" class="data row0 col4" >0.000000</td>
      <td id="T_c8b0a_row0_col5" class="data row0 col5" >0.000000</td>
      <td id="T_c8b0a_row0_col6" class="data row0 col6" >91.000000</td>
      <td id="T_c8b0a_row0_col7" class="data row0 col7" >1.000000</td>
      <td id="T_c8b0a_row0_col8" class="data row0 col8" >875.000000</td>
      <td id="T_c8b0a_row0_col9" class="data row0 col9" >1499.000000</td>
      <td id="T_c8b0a_row0_col10" class="data row0 col10" >2372.000000</td>
      <td id="T_c8b0a_row0_col11" class="data row0 col11" >3454.000000</td>
      <td id="T_c8b0a_row0_col12" class="data row0 col12" >5790.000000</td>
      <td id="T_c8b0a_row0_col13" class="data row0 col13" >2883.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row1" class="row_heading level0 row1" >2</th>
      <td id="T_c8b0a_row1_col0" class="data row1 col0" >1.000000</td>
      <td id="T_c8b0a_row1_col1" class="data row1 col1" >0.000000</td>
      <td id="T_c8b0a_row1_col2" class="data row1 col2" >0.000000</td>
      <td id="T_c8b0a_row1_col3" class="data row1 col3" >0.000000</td>
      <td id="T_c8b0a_row1_col4" class="data row1 col4" >2.000000</td>
      <td id="T_c8b0a_row1_col5" class="data row1 col5" >0.000000</td>
      <td id="T_c8b0a_row1_col6" class="data row1 col6" >50.000000</td>
      <td id="T_c8b0a_row1_col7" class="data row1 col7" >44.000000</td>
      <td id="T_c8b0a_row1_col8" class="data row1 col8" >241.000000</td>
      <td id="T_c8b0a_row1_col9" class="data row1 col9" >1651.000000</td>
      <td id="T_c8b0a_row1_col10" class="data row1 col10" >1735.000000</td>
      <td id="T_c8b0a_row1_col11" class="data row1 col11" >2870.000000</td>
      <td id="T_c8b0a_row1_col12" class="data row1 col12" >2685.000000</td>
      <td id="T_c8b0a_row1_col13" class="data row1 col13" >2572.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row2" class="row_heading level0 row2" >3</th>
      <td id="T_c8b0a_row2_col0" class="data row2 col0" >0.000000</td>
      <td id="T_c8b0a_row2_col1" class="data row2 col1" >0.000000</td>
      <td id="T_c8b0a_row2_col2" class="data row2 col2" >0.000000</td>
      <td id="T_c8b0a_row2_col3" class="data row2 col3" >0.000000</td>
      <td id="T_c8b0a_row2_col4" class="data row2 col4" >0.000000</td>
      <td id="T_c8b0a_row2_col5" class="data row2 col5" >30.000000</td>
      <td id="T_c8b0a_row2_col6" class="data row2 col6" >0.000000</td>
      <td id="T_c8b0a_row2_col7" class="data row2 col7" >50.000000</td>
      <td id="T_c8b0a_row2_col8" class="data row2 col8" >283.000000</td>
      <td id="T_c8b0a_row2_col9" class="data row2 col9" >2135.000000</td>
      <td id="T_c8b0a_row2_col10" class="data row2 col10" >3537.000000</td>
      <td id="T_c8b0a_row2_col11" class="data row2 col11" >4060.000000</td>
      <td id="T_c8b0a_row2_col12" class="data row2 col12" >2712.000000</td>
      <td id="T_c8b0a_row2_col13" class="data row2 col13" >2396.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row3" class="row_heading level0 row3" >4</th>
      <td id="T_c8b0a_row3_col0" class="data row3 col0" >0.000000</td>
      <td id="T_c8b0a_row3_col1" class="data row3 col1" >0.000000</td>
      <td id="T_c8b0a_row3_col2" class="data row3 col2" >0.000000</td>
      <td id="T_c8b0a_row3_col3" class="data row3 col3" >0.000000</td>
      <td id="T_c8b0a_row3_col4" class="data row3 col4" >0.000000</td>
      <td id="T_c8b0a_row3_col5" class="data row3 col5" >0.000000</td>
      <td id="T_c8b0a_row3_col6" class="data row3 col6" >21.000000</td>
      <td id="T_c8b0a_row3_col7" class="data row3 col7" >37.000000</td>
      <td id="T_c8b0a_row3_col8" class="data row3 col8" >368.000000</td>
      <td id="T_c8b0a_row3_col9" class="data row3 col9" >1436.000000</td>
      <td id="T_c8b0a_row3_col10" class="data row3 col10" >2295.000000</td>
      <td id="T_c8b0a_row3_col11" class="data row3 col11" >3258.000000</td>
      <td id="T_c8b0a_row3_col12" class="data row3 col12" >3999.000000</td>
      <td id="T_c8b0a_row3_col13" class="data row3 col13" >4058.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row4" class="row_heading level0 row4" >5</th>
      <td id="T_c8b0a_row4_col0" class="data row4 col0" >0.000000</td>
      <td id="T_c8b0a_row4_col1" class="data row4 col1" >24.000000</td>
      <td id="T_c8b0a_row4_col2" class="data row4 col2" >0.000000</td>
      <td id="T_c8b0a_row4_col3" class="data row4 col3" >72.000000</td>
      <td id="T_c8b0a_row4_col4" class="data row4 col4" >0.000000</td>
      <td id="T_c8b0a_row4_col5" class="data row4 col5" >0.000000</td>
      <td id="T_c8b0a_row4_col6" class="data row4 col6" >0.000000</td>
      <td id="T_c8b0a_row4_col7" class="data row4 col7" >74.000000</td>
      <td id="T_c8b0a_row4_col8" class="data row4 col8" >273.000000</td>
      <td id="T_c8b0a_row4_col9" class="data row4 col9" >1618.000000</td>
      <td id="T_c8b0a_row4_col10" class="data row4 col10" >1746.000000</td>
      <td id="T_c8b0a_row4_col11" class="data row4 col11" >2525.000000</td>
      <td id="T_c8b0a_row4_col12" class="data row4 col12" >3346.000000</td>
      <td id="T_c8b0a_row4_col13" class="data row4 col13" >2936.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row5" class="row_heading level0 row5" >6</th>
      <td id="T_c8b0a_row5_col0" class="data row5 col0" >0.000000</td>
      <td id="T_c8b0a_row5_col1" class="data row5 col1" >0.000000</td>
      <td id="T_c8b0a_row5_col2" class="data row5 col2" >0.000000</td>
      <td id="T_c8b0a_row5_col3" class="data row5 col3" >0.000000</td>
      <td id="T_c8b0a_row5_col4" class="data row5 col4" >0.000000</td>
      <td id="T_c8b0a_row5_col5" class="data row5 col5" >0.000000</td>
      <td id="T_c8b0a_row5_col6" class="data row5 col6" >19.000000</td>
      <td id="T_c8b0a_row5_col7" class="data row5 col7" >101.000000</td>
      <td id="T_c8b0a_row5_col8" class="data row5 col8" >231.000000</td>
      <td id="T_c8b0a_row5_col9" class="data row5 col9" >2004.000000</td>
      <td id="T_c8b0a_row5_col10" class="data row5 col10" >1476.000000</td>
      <td id="T_c8b0a_row5_col11" class="data row5 col11" >3256.000000</td>
      <td id="T_c8b0a_row5_col12" class="data row5 col12" >3927.000000</td>
      <td id="T_c8b0a_row5_col13" class="data row5 col13" >4399.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row6" class="row_heading level0 row6" >7</th>
      <td id="T_c8b0a_row6_col0" class="data row6 col0" >0.000000</td>
      <td id="T_c8b0a_row6_col1" class="data row6 col1" >0.000000</td>
      <td id="T_c8b0a_row6_col2" class="data row6 col2" >0.000000</td>
      <td id="T_c8b0a_row6_col3" class="data row6 col3" >0.000000</td>
      <td id="T_c8b0a_row6_col4" class="data row6 col4" >0.000000</td>
      <td id="T_c8b0a_row6_col5" class="data row6 col5" >0.000000</td>
      <td id="T_c8b0a_row6_col6" class="data row6 col6" >4.000000</td>
      <td id="T_c8b0a_row6_col7" class="data row6 col7" >92.000000</td>
      <td id="T_c8b0a_row6_col8" class="data row6 col8" >671.000000</td>
      <td id="T_c8b0a_row6_col9" class="data row6 col9" >1438.000000</td>
      <td id="T_c8b0a_row6_col10" class="data row6 col10" >3456.000000</td>
      <td id="T_c8b0a_row6_col11" class="data row6 col11" >2883.000000</td>
      <td id="T_c8b0a_row6_col12" class="data row6 col12" >3058.000000</td>
      <td id="T_c8b0a_row6_col13" class="data row6 col13" >6400.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row7" class="row_heading level0 row7" >8</th>
      <td id="T_c8b0a_row7_col0" class="data row7 col0" >0.000000</td>
      <td id="T_c8b0a_row7_col1" class="data row7 col1" >0.000000</td>
      <td id="T_c8b0a_row7_col2" class="data row7 col2" >0.000000</td>
      <td id="T_c8b0a_row7_col3" class="data row7 col3" >0.000000</td>
      <td id="T_c8b0a_row7_col4" class="data row7 col4" >0.000000</td>
      <td id="T_c8b0a_row7_col5" class="data row7 col5" >30.000000</td>
      <td id="T_c8b0a_row7_col6" class="data row7 col6" >18.000000</td>
      <td id="T_c8b0a_row7_col7" class="data row7 col7" >45.000000</td>
      <td id="T_c8b0a_row7_col8" class="data row7 col8" >372.000000</td>
      <td id="T_c8b0a_row7_col9" class="data row7 col9" >2302.000000</td>
      <td id="T_c8b0a_row7_col10" class="data row7 col10" >3508.000000</td>
      <td id="T_c8b0a_row7_col11" class="data row7 col11" >2808.000000</td>
      <td id="T_c8b0a_row7_col12" class="data row7 col12" >2671.000000</td>
      <td id="T_c8b0a_row7_col13" class="data row7 col13" >4093.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row8" class="row_heading level0 row8" >9</th>
      <td id="T_c8b0a_row8_col0" class="data row8 col0" >0.000000</td>
      <td id="T_c8b0a_row8_col1" class="data row8 col1" >0.000000</td>
      <td id="T_c8b0a_row8_col2" class="data row8 col2" >0.000000</td>
      <td id="T_c8b0a_row8_col3" class="data row8 col3" >135.000000</td>
      <td id="T_c8b0a_row8_col4" class="data row8 col4" >0.000000</td>
      <td id="T_c8b0a_row8_col5" class="data row8 col5" >49.000000</td>
      <td id="T_c8b0a_row8_col6" class="data row8 col6" >1.000000</td>
      <td id="T_c8b0a_row8_col7" class="data row8 col7" >133.000000</td>
      <td id="T_c8b0a_row8_col8" class="data row8 col8" >689.000000</td>
      <td id="T_c8b0a_row8_col9" class="data row8 col9" >2428.000000</td>
      <td id="T_c8b0a_row8_col10" class="data row8 col10" >2652.000000</td>
      <td id="T_c8b0a_row8_col11" class="data row8 col11" >2615.000000</td>
      <td id="T_c8b0a_row8_col12" class="data row8 col12" >3852.000000</td>
      <td id="T_c8b0a_row8_col13" class="data row8 col13" >4270.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row9" class="row_heading level0 row9" >10</th>
      <td id="T_c8b0a_row9_col0" class="data row9 col0" >0.000000</td>
      <td id="T_c8b0a_row9_col1" class="data row9 col1" >0.000000</td>
      <td id="T_c8b0a_row9_col2" class="data row9 col2" >0.000000</td>
      <td id="T_c8b0a_row9_col3" class="data row9 col3" >96.000000</td>
      <td id="T_c8b0a_row9_col4" class="data row9 col4" >0.000000</td>
      <td id="T_c8b0a_row9_col5" class="data row9 col5" >25.000000</td>
      <td id="T_c8b0a_row9_col6" class="data row9 col6" >19.000000</td>
      <td id="T_c8b0a_row9_col7" class="data row9 col7" >270.000000</td>
      <td id="T_c8b0a_row9_col8" class="data row9 col8" >1154.000000</td>
      <td id="T_c8b0a_row9_col9" class="data row9 col9" >2598.000000</td>
      <td id="T_c8b0a_row9_col10" class="data row9 col10" >3850.000000</td>
      <td id="T_c8b0a_row9_col11" class="data row9 col11" >4157.000000</td>
      <td id="T_c8b0a_row9_col12" class="data row9 col12" >3965.000000</td>
      <td id="T_c8b0a_row9_col13" class="data row9 col13" >0.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row10" class="row_heading level0 row10" >11</th>
      <td id="T_c8b0a_row10_col0" class="data row10 col0" >0.000000</td>
      <td id="T_c8b0a_row10_col1" class="data row10 col1" >6.000000</td>
      <td id="T_c8b0a_row10_col2" class="data row10 col2" >20.000000</td>
      <td id="T_c8b0a_row10_col3" class="data row10 col3" >0.000000</td>
      <td id="T_c8b0a_row10_col4" class="data row10 col4" >16.000000</td>
      <td id="T_c8b0a_row10_col5" class="data row10 col5" >9.000000</td>
      <td id="T_c8b0a_row10_col6" class="data row10 col6" >47.000000</td>
      <td id="T_c8b0a_row10_col7" class="data row10 col7" >36.000000</td>
      <td id="T_c8b0a_row10_col8" class="data row10 col8" >632.000000</td>
      <td id="T_c8b0a_row10_col9" class="data row10 col9" >1430.000000</td>
      <td id="T_c8b0a_row10_col10" class="data row10 col10" >3195.000000</td>
      <td id="T_c8b0a_row10_col11" class="data row10 col11" >6084.000000</td>
      <td id="T_c8b0a_row10_col12" class="data row10 col12" >3390.000000</td>
      <td id="T_c8b0a_row10_col13" class="data row10 col13" >0.000000</td>
    </tr>
    <tr>
      <th id="T_c8b0a_level0_row11" class="row_heading level0 row11" >12</th>
      <td id="T_c8b0a_row11_col0" class="data row11 col0" >0.000000</td>
      <td id="T_c8b0a_row11_col1" class="data row11 col1" >0.000000</td>
      <td id="T_c8b0a_row11_col2" class="data row11 col2" >0.000000</td>
      <td id="T_c8b0a_row11_col3" class="data row11 col3" >0.000000</td>
      <td id="T_c8b0a_row11_col4" class="data row11 col4" >18.000000</td>
      <td id="T_c8b0a_row11_col5" class="data row11 col5" >63.000000</td>
      <td id="T_c8b0a_row11_col6" class="data row11 col6" >94.000000</td>
      <td id="T_c8b0a_row11_col7" class="data row11 col7" >337.000000</td>
      <td id="T_c8b0a_row11_col8" class="data row11 col8" >1801.000000</td>
      <td id="T_c8b0a_row11_col9" class="data row11 col9" >2216.000000</td>
      <td id="T_c8b0a_row11_col10" class="data row11 col10" >3576.000000</td>
      <td id="T_c8b0a_row11_col11" class="data row11 col11" >5156.000000</td>
      <td id="T_c8b0a_row11_col12" class="data row11 col12" >3628.000000</td>
      <td id="T_c8b0a_row11_col13" class="data row11 col13" >0.000000</td>
    </tr>
  </tbody>
</table>




The table above shows the amount of netflix releases by month and year added. 
It seems that most Netflix TV shows and movies are typically more released in the second half of the year.

## 3.3 Which country has most Netflix content?

Netflix offers internationally diverse content. Now, let's take a look at the countries in which Netflix contents are produced. 


```python
# Top 10 countries
country_filtered = netflix_new.loc[netflix_new["Country"]!="Unavailable",]
countries = country_filtered.groupby(["Country"])["title"].nunique().reset_index()
countries = countries.sort_values(by="title", ascending=False).reset_index()
countries = countries.drop("index", axis=1)
countries.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>United States</td>
      <td>3680</td>
    </tr>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>1046</td>
    </tr>
    <tr>
      <th>2</th>
      <td>United Kingdom</td>
      <td>803</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Canada</td>
      <td>445</td>
    </tr>
    <tr>
      <th>4</th>
      <td>France</td>
      <td>393</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Japan</td>
      <td>316</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Spain</td>
      <td>232</td>
    </tr>
    <tr>
      <th>7</th>
      <td>South Korea</td>
      <td>231</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Germany</td>
      <td>226</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Mexico</td>
      <td>169</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Draw a barplot
countries_plt = sns.barplot(countries.head(10), y="Country", x="title", palette="rocket")
countries_plt.set_title("Top 10 countries that produce the most content on Netflix")
countries_plt.set_ylabel("")
sns.set(font_scale=1.9)
matplotlib.style.use("seaborn-white")
```

    C:\Users\eunbi\AppData\Local\Temp\ipykernel_19832\1056808559.py:6: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      matplotlib.style.use("seaborn-white")
    


    
![png](output_44_1.png)
    


The majority of Netflix contents obviously come from United States. The second biggest contributor is India. The two countries with the largest film industry, Hollywood and Bollywood.


```python
# Top five countries
topfive_countries = countries["Country"].head().tolist()
topfive_countries_df = country_filtered.loc[country_filtered["Country"].isin(topfive_countries),]
# Group by 'year_added' & 'country' and count the number of titles
year_country = pd.pivot_table(topfive_countries_df, index="year_added", columns = "Country", values = "title", aggfunc=pd.Series.nunique).reset_index()
year_country.columns = ["year_added", "Canada", "France", "India", "United Kingdom", "United States"]
year_country = year_country.fillna(0)
year_country.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year_added</th>
      <th>Canada</th>
      <th>France</th>
      <th>India</th>
      <th>United Kingdom</th>
      <th>United States</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2008</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Draw a line chart
year_plt = year_country[:-1].plot.line(x="year_added", figsize=(15,8))
year_plt.set_title("Top 5 countries across the years ~(2020)")
sns.set(font_scale=1.8)
matplotlib.style.use("seaborn-white")
```

    C:\Users\eunbi\AppData\Local\Temp\ipykernel_19832\3916927087.py:5: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      matplotlib.style.use("seaborn-white")
    


    
![png](output_47_1.png)
    


The gap between the number of American contents and the number of Indian contents started to widen as the number of Indian contents significantly dropped in 2018. The number of UK contents started to drop, too. We can also see that the number of Japanese and French contents kept slightly increasing, suggesting that the demand for international contents has increased in recent years.

## 3.4 Who has made the most movies on Netflix?


```python
# Top 10 directors
director_filtered = netflix_new.loc[netflix_new["director"]!="Unavailable",]
directors = director_filtered.groupby(["director"])["title"].nunique().reset_index()
directors = directors.sort_values(by="title", ascending=False).reset_index()
directors = directors.drop("index", axis=1)
directors.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>director</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rajiv Chilaka</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Raúl Campos, Jan Suter</td>
      <td>18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Suhas Kadav</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Marcus Raboy</td>
      <td>16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jay Karas</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Cathy Garcia-Molina</td>
      <td>13</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Jay Chapman</td>
      <td>12</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Youssef Chahine</td>
      <td>12</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Martin Scorsese</td>
      <td>12</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Steven Spielberg</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Draw a barplot
directors_plt = sns.barplot(directors.head(10), y="director", x="title", palette="Blues_r")
directors_plt.set_title("Top 10 Directors that produce the most content on Netflix")
sns.set(font_scale=1.9)
directors_plt.set_ylabel("")
matplotlib.style.use("seaborn-white")
```

    C:\Users\eunbi\AppData\Local\Temp\ipykernel_19832\925380534.py:6: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      matplotlib.style.use("seaborn-white")
    


    
![png](output_51_1.png)
    


The directors on Netflix are internationally diverse. The most popular director, based on the number of titles, is Rajiv Chilaka. 

## 3.5 What is the top genre on Netflix?
I want to explore what the most common genres on Netflix are.


```python
# Top 10 genres
genres = netflix_new.groupby(["Genre"])["title"].nunique().reset_index()
genres = genres.sort_values(by="title", ascending=False).reset_index()
genres = genres.drop("index", axis=1)
genres.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>International Movies</td>
      <td>2752</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dramas</td>
      <td>2426</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Comedies</td>
      <td>1674</td>
    </tr>
    <tr>
      <th>3</th>
      <td>International TV Shows</td>
      <td>1349</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Documentaries</td>
      <td>869</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Action &amp; Adventure</td>
      <td>859</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TV Dramas</td>
      <td>762</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Independent Movies</td>
      <td>756</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Children &amp; Family Movies</td>
      <td>641</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Romantic Movies</td>
      <td>616</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Draw a barplot
genres_plt = sns.barplot(genres.head(10), y="Genre", x="title", palette="Reds_r")
genres_plt.set_title("Top 10 Genres on Netflix")
sns.set(font_scale=1.9)
genres_plt.set_ylabel("")
matplotlib.style.use("seaborn-white")
```

    C:\Users\eunbi\AppData\Local\Temp\ipykernel_19832\1746428977.py:6: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      matplotlib.style.use("seaborn-white")
    


    
![png](output_55_1.png)
    


The most common genre on Netflix is International Movies. The next most common genres are Dramas and Comedies. I wonder if there would be a difference between the movies and TV shows in terms of common genres.


```python
movie_tv_genres = netflix_new.groupby(["type", "Genre"])["title"].nunique().reset_index()
movie_tv_genres = movie_tv_genres.sort_values(by=["type", "title"], ascending=False)
movie_tv_genres.loc[movie_tv_genres["Genre"].str.contains("International"), "Genre"] = "International"
movie_tv_genres.loc[movie_tv_genres["Genre"].str.contains("Dramas"), "Genre"] = "Dramas"
movie_tv_genres.loc[movie_tv_genres["Genre"].str.contains("Comedies"), "Genre"] = "Comedies"
movie_tv_genres.loc[movie_tv_genres["Genre"].str.contains("Crime"), "Genre"] = "Crime"
movie_tv_genres.loc[movie_tv_genres["Genre"].str.contains("Kids"), "Genre"] = "Kids"
movie_tv_genres.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>Genre</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>TV Show</td>
      <td>International</td>
      <td>1349</td>
    </tr>
    <tr>
      <th>35</th>
      <td>TV Show</td>
      <td>Dramas</td>
      <td>762</td>
    </tr>
    <tr>
      <th>34</th>
      <td>TV Show</td>
      <td>Comedies</td>
      <td>573</td>
    </tr>
    <tr>
      <th>23</th>
      <td>TV Show</td>
      <td>Crime</td>
      <td>469</td>
    </tr>
    <tr>
      <th>26</th>
      <td>TV Show</td>
      <td>Kids</td>
      <td>448</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Draw a barplot
palette ={"International": "C0", "Dramas": "C1", "Comedies": "C2", "Documentaries": "C3", 
          "Action & Adventure": "C4", "Crime" : "C5", "Kids" : "C6"}
fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
sns.barplot(movie_tv_genres.loc[movie_tv_genres["type"]=="Movie"].head(5), ax=axes[0], x="Genre", y="title", palette=palette)
axes[0].set_title("Top Five Movie Genres")
axes[0].tick_params(axis='x', rotation=20)
sns.barplot(movie_tv_genres.loc[movie_tv_genres["type"]=="TV Show"].head(5), ax=axes[1], x="Genre", y="title", palette=palette)
axes[1].set_title("Top Five TV Genres")
axes[1].tick_params(axis='x', rotation=20)
sns.set(font_scale=1.5)
matplotlib.style.use("seaborn-white")
```

    C:\Users\eunbi\AppData\Local\Temp\ipykernel_19832\3348756340.py:12: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      matplotlib.style.use("seaborn-white")
    


    
![png](output_58_1.png)
    


The common genres of movies and TV shows seem to be similar. For both movies and TV shows, International, dramas and comedies are the most common genres. 

## 3.6 What country mostly produces what genre?

Now, I wonder if certain genres of contents come from certain countries. For example, where do dramas mostly come from?


```python
# ceate a pivot table by country and genre
topfive_genres = genres["Genre"].head(5).tolist()
country_genre = netflix_new.loc[(netflix_new["Genre"].isin(topfive_genres))&
                                (netflix_new["Country"]!="Unavailable")].groupby(["Country", "Genre"])["title"].nunique().reset_index()
country_genre = country_genre.sort_values(by=["Country", "title"], ascending=False)
country_genre = country_genre.loc[country_genre["Country"].isin(topfive_countries)].groupby(["Country"]).head(5)
country_genre.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Genre</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>378</th>
      <td>United States</td>
      <td>Dramas</td>
      <td>835</td>
    </tr>
    <tr>
      <th>376</th>
      <td>United States</td>
      <td>Comedies</td>
      <td>680</td>
    </tr>
    <tr>
      <th>377</th>
      <td>United States</td>
      <td>Documentaries</td>
      <td>511</td>
    </tr>
    <tr>
      <th>379</th>
      <td>United States</td>
      <td>International Movies</td>
      <td>166</td>
    </tr>
    <tr>
      <th>380</th>
      <td>United States</td>
      <td>International TV Shows</td>
      <td>73</td>
    </tr>
    <tr>
      <th>370</th>
      <td>United Kingdom</td>
      <td>Dramas</td>
      <td>196</td>
    </tr>
    <tr>
      <th>371</th>
      <td>United Kingdom</td>
      <td>International Movies</td>
      <td>168</td>
    </tr>
    <tr>
      <th>372</th>
      <td>United Kingdom</td>
      <td>International TV Shows</td>
      <td>128</td>
    </tr>
    <tr>
      <th>369</th>
      <td>United Kingdom</td>
      <td>Documentaries</td>
      <td>127</td>
    </tr>
    <tr>
      <th>368</th>
      <td>United Kingdom</td>
      <td>Comedies</td>
      <td>91</td>
    </tr>
  </tbody>
</table>
</div>




```python
# draw a grouped  bar plot
country_genre_plt = sns.barplot(country_genre, x="Genre", y="title", hue="Country")
country_genre_plt.set_title("What genres do countries mostly produce?")
sns.set(font_scale=1.5)
matplotlib.style.use("seaborn-white")
```

    C:\Users\eunbi\AppData\Local\Temp\ipykernel_19832\833652562.py:5: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      matplotlib.style.use("seaborn-white")
    


    
![png](output_62_1.png)
    


Again, we can see that the US and India are the major content contributors regardless of genre. As for Dramas and Comedies, the first and second most popular genres on Netflix, the majority of the contents comes from the US and India. The major producer of Documentaries is the US, followed by the UK being the second biggest producer. We can see that India is the least producer of Documentaries, while it is the biggest producer of International movies. Lastly, as for International TV shows, the UK is the largest producer. 

## 3.7 How are the ratings distributed?

I will group the data by 'type' and 'rating' to see how the ratings are distributed based on the number of titles.


```python
# create a pivot table by rating and type
ratings = pd.pivot_table(netflix_new, index="rating", columns = "type", values = "title", 
                         aggfunc=pd.Series.nunique).reset_index()
ratings = ratings.fillna(0)
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>type</th>
      <th>rating</th>
      <th>Movie</th>
      <th>TV Show</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>G</td>
      <td>41.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NC-17</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NR</td>
      <td>75.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PG</td>
      <td>287.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PG-13</td>
      <td>490.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a stacked bar chart
sns.set(style='white')
sns.set(font_scale=1.5)
ratings.set_index("rating").plot(kind='bar', stacked=True, color=['red', 'black'], figsize=(16, 8))
plt.title("Amount of Netflix content by rating")
plt.xticks(rotation=0)
plt.xlabel("Rating")
```




    Text(0.5, 0, 'Rating')




    
![png](output_66_1.png)
    


The largest count of Netflix content is rated “TV-MA". TV-MA” is a rating assigned by the TV Parental Guidelines to a television program designed for mature audiences only. The second largest count is rated “TV-14". “TV-14” contains material that parents or adult guardians may find unsuitable for children under the age of 14. The target audience of Netflix are mainly young adults and older adults.

## 3.8 Who are the most popular actors on Netflix?

Some actors may star more frequently on Netflix contents than others. Now, I want to explore the top Actor on Netflix based on the number of titles.


```python
# Top 10 actors
actors = netflix_new.loc[netflix_new["Actors"]!="Unavailable"].groupby(["Actors"])["title"].nunique().reset_index()
actors = actors.sort_values(by="title", ascending=False).reset_index()
actors = actors.drop("index", axis=1)
actors.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actors</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Anupam Kher</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Shah Rukh Khan</td>
      <td>35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Julie Tejwani</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Takahiro Sakurai</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Naseeruddin Shah</td>
      <td>32</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Rupa Bhimani</td>
      <td>31</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Akshay Kumar</td>
      <td>30</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Om Puri</td>
      <td>30</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Yuki Kaji</td>
      <td>29</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Paresh Rawal</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Draw a barplot
actors_plt = sns.barplot(actors.head(10), y="Actors", x="title", palette="pastel")
actors_plt.set_title("Top 10 Actors on Netflix")
actors_plt.set_ylabel("")
sns.set(font_scale=1.9)
matplotlib.style.use("seaborn-white")
```

    C:\Users\eunbi\AppData\Local\Temp\ipykernel_19832\3620370062.py:6: MatplotlibDeprecationWarning: The seaborn styles shipped by Matplotlib are deprecated since 3.6, as they no longer correspond to the styles shipped by seaborn. However, they will remain available as 'seaborn-v0_8-<style>'. Alternatively, directly use the seaborn API instead.
      matplotlib.style.use("seaborn-white")
    


    
![png](output_70_1.png)
    


We can see that many indian actors appear on Netflix contents. The most popular actor on Netflix is an Indian actor, Anupam Kher. The next most popular actors are Shah Rukh Khan and Julie Tejwani.

## 3.9 What words mostly appear on Netflix description?

The final thing I want to do is the text analysis of Netflix content descriptions. To do this, I will start by extracting "description" column from the data and converting it to a string.


```python
description = netflix["description"].tolist()
description =str(description)
description = re.sub("[\"\-\[\]\'\.\?]", "", description)
description = description.replace(" ", ",")
description = re.sub(",{2,}", ",", description)
description[:1000]
```




    'As,her,father,nears,the,end,of,his,life,filmmaker,Kirsten,Johnson,stages,his,death,in,inventive,and,comical,ways,to,help,them,both,face,the,inevitable,After,crossing,paths,at,a,party,a,Cape,Town,teen,sets,out,to,prove,whether,a,privateschool,swimming,star,is,her,sister,who,was,abducted,at,birth,To,protect,his,family,from,a,powerful,drug,lord,skilled,thief,Mehdi,and,his,expert,team,of,robbers,are,pulled,into,a,violent,and,deadly,turf,war,Feuds,flirtations,and,toilet,talk,go,down,among,the,incarcerated,women,at,the,Orleans,Justice,Center,in,New,Orleans,on,this,gritty,reality,series,In,a,city,of,coaching,centers,known,to,train,India’s,finest,collegiate,minds,an,earnest,but,unexceptional,student,and,his,friends,navigate,campus,life,The,arrival,of,a,charismatic,young,priest,brings,glorious,miracles,ominous,mysteries,and,renewed,religious,fervor,to,a,dying,town,desperate,to,believe,Equestrias,divided,But,a,brighteyed,hero,believes,Earth,Ponies,Pegasi,and,Unicorns,should,be,pals,—,and,hoof,to'



Now, I will create a wordcloud using WordCloud() function from wordcloud package. 


```python
from wordcloud import WordCloud
from wordcloud import STOPWORDS
def wordcloud(data, width=1200, height=500):
    word_draw = WordCloud(
        font_path=r"C:\Windows\Fonts\Verdana.ttf",
        stopwords=STOPWORDS,
        width=width, height=height, 
        background_color="white",
        random_state=42
    )
    word_draw.generate(data)

    plt.figure(figsize=(16, 8))
    plt.imshow(word_draw)
    plt.axis("off")
    plt.show()
```


```python
wordcloud(description, width=800, height=500)
```


    
![png](output_76_0.png)
    


Looking at the wordcloud above, the words 'find', 'life' and 'family' seem to be the most frequent words in the content descriptions. The words 'new', 'take', 'world' and 'love' also frequently appear.

# 4. Conclusions

We have drawn many interesting inferences from the dataset. Here’s a summary of my analysis.

1. The majority content type on Netflix is movie.
2. Netflix started gaining traction after 2014. Since then, the amount of content added has been increasing significantly.
3. The country that produces the most amount of content is the United States.
4. The directors on Netflix is mainly international. The most popular director on Netflix, with the most titles, is Rajiv Chilaka.
5. The most popular genre on Netflix is International Movies.
6. The US and India are the major content contributors regardless of genre. 
7. The largest count of Netflix content is made with “TV-MA” rating as for both movies and TV shows.
8. The actors on Netflix is mainly international. The most popular actor on Netflix TV Shows, with the most titles, is Anupam Kher.
9. The most frequent words in the netflix content descriptions are about family, life and world.
