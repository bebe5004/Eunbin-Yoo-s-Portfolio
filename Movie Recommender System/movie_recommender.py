# import libraries
import streamlit as st
import pickle


# import files
movies = pickle.load(open('movies.pkl', 'rb'))
movies_df = pickle.load(open('movies_df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies_list = movies['title'].values

# create title for stream lit page
st.title("""Movie Recommender System
This is a content-based movie recommender system based on features of movies :smile: """)


# create a input box for a movie name
selected_movie = st.selectbox('What is your favorite movie?', movies_list)


# movie recommender algorithm
def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    sorted_movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in sorted_movie_list:
        poster_path = movies["poster_path"][i[0]]
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append("https://image.tmdb.org/t/p/original"+poster_path)

    return recommended_movies, recommended_posters

# details of movies
movie_info = ["title", "genres", "overview", "release_date", "credits", "original_language"]
mv_dataframe = movies_df[movie_info]

# create a recommend button with function of displaying recommended movies and movie posters
if st.button('Show recommendations'):
    recommendation, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write(recommendation[0])
        st.image(posters[0])
        st.write("Genre: " + mv_dataframe[mv_dataframe['title'] == recommendation[0]].genres.values[0])
        st.write("Release date: " + mv_dataframe[mv_dataframe['title'] == recommendation[0]].release_date.values[0])
        st.write("Language: " + mv_dataframe[mv_dataframe['title'] == recommendation[0]].original_language.values[0])
        st.write("Overview: " + mv_dataframe[mv_dataframe['title'] == recommendation[0]].overview.values[0])

    with col2:
        st.write(recommendation[1])
        st.image(posters[1])
        st.write("Genre: " + mv_dataframe[mv_dataframe['title'] == recommendation[1]].genres.values[0])
        st.write("Release date: " + mv_dataframe[mv_dataframe['title'] == recommendation[1]].release_date.values[0])
        st.write("Language: " + mv_dataframe[mv_dataframe['title'] == recommendation[1]].original_language.values[0])
        st.write("Overview: " + mv_dataframe[mv_dataframe['title'] == recommendation[1]].overview.values[0])

    with col3:
        st.write(recommendation[2])
        st.image(posters[2])
        st.write("Genre: " + mv_dataframe[mv_dataframe['title'] == recommendation[2]].genres.values[0])
        st.write("Release date: " + mv_dataframe[mv_dataframe['title'] == recommendation[2]].release_date.values[0])
        st.write("Language: " + mv_dataframe[mv_dataframe['title'] == recommendation[2]].original_language.values[0])
        st.write("Overview: " + mv_dataframe[mv_dataframe['title'] == recommendation[2]].overview.values[0])

    with col4:
        st.write(recommendation[3])
        st.image(posters[3])
        st.write("Genre: " + mv_dataframe[mv_dataframe['title'] == recommendation[3]].genres.values[0])
        st.write("Release date: " + mv_dataframe[mv_dataframe['title'] == recommendation[3]].release_date.values[0])
        st.write("Language: " + mv_dataframe[mv_dataframe['title'] == recommendation[3]].original_language.values[0])
        st.write("Overview: " + mv_dataframe[mv_dataframe['title'] == recommendation[3]].overview.values[0])

    with col5:
        st.write(recommendation[4])
        st.image(posters[4])
        st.write("Genre: " + mv_dataframe[mv_dataframe['title'] == recommendation[4]].genres.values[0])
        st.write("Release date: " + mv_dataframe[mv_dataframe['title'] == recommendation[4]].release_date.values[0])
        st.write("Language: " + mv_dataframe[mv_dataframe['title'] == recommendation[4]].original_language.values[0])
        st.write("Overview: " + mv_dataframe[mv_dataframe['title'] == recommendation[4]].overview.values[0])

