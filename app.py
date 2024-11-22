import streamlit as st
import pandas as pd
import requests

def fetch_poster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=2e3c18dcd8ccad87d3b2753ea1158a08'
    response = requests.get(url)
    data = response.json()
    return f"https://image.tmdb.org/t/p/w780{data['poster_path']}"

df = pd.read_pickle(r'C:\Users\Vidya\Downloads\Recommendation System\movies.pkl')
movie_list = df['title'].tolist()
similarity = pd.read_pickle(r'C:\Users\Vidya\Downloads\Recommendation System\similarity.pkl')
recom = []
posters = []
def recommend(movie):
    movie_index = df[df['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse = True, key=lambda x: x[1])[1:6]
    for i in movie_list:
        recom.append(df.iloc[i[0]].title)
        posters.append(fetch_poster(df['id'].iloc[i[0]]))
    return recom, posters
st.title('Movie Recommender System')
option = st.selectbox('Please select a movie name',movie_list)
if st.button('Recommend'):
    recommendations, posters = recommend(option)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(posters[0])
        st.markdown(recommendations[0])
    with col2:
        st.image(posters[1])
        st.markdown(recommendations[1])
    with col3:
        st.image(posters[2])
        st.markdown(recommendations[2])
    with col4:
        st.image(posters[3])
        st.markdown(recommendations[3])
    with col5:
        st.image(posters[4])
        st.markdown(recommendations[4])