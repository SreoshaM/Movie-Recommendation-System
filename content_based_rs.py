"""Recommandation systems are of three types mainly:
1) Content Based : Where we classify depending on type of contents. 
                   For example, Genre of songs, where different Genre values are the different contents. 
2) Collaborative Filtering : Where we focus on the user choice, by considering the fact that if A and B user both are 
                        liking P1 product and if A user also liked P2 product, then B would like P2 too ( Transitive Property).
3) Hybrid : This is a combined scheme of both Content Based and Collaborative Filtering."""
import ast
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class MRS:
    def __init__(self):
        self.ps = PorterStemmer()
        self.cv = CountVectorizer(max_features = 5000, stop_words='english')

    def convert(self, obj):
        l = []
        for i in ast.literal_eval(obj):
            l.append(i['name'])
        return l
    
    def convert_cast(self, obj):
        l = []
        count = 0
        for i in ast.literal_eval(obj):
            if count<=2:
                l.append(i['name'])
                count+=1
        return l
    
    def fetch_director(self, obj):
        l = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                l.append(i['name'])
        return l
    
    def stem(self,text):
        y = []
        for i in text.split():
            y.append(self.ps.stem(i))
        return " ".join(y)

    def preparation(self, movie_path, credit_path):

        movies = pd.read_csv(movie_path)
        credits = pd.read_csv(credit_path)
        movies = movies.merge(credits, on = 'title')
        movies = movies[['genres', 'id', 'keywords', 'title', 'overview', 'release_date', 'cast', 'crew']]
        movies.dropna(inplace = True)
        movies['genres'] = movies['genres'].apply(self.convert)
        movies['keywords'] = movies['keywords'].apply(self.convert)
        movies['cast'] = movies['cast'].apply(self.convert_cast)
        movies['crew'] = movies['crew'].apply(self.fetch_director)
        movies['overview'] = movies['overview'].apply(lambda row: row.split(","))
        movies['genres'] = movies['genres'].apply(lambda row: [x.replace(" ", "") for x in row])
        movies['keywords'] = movies['keywords'].apply(lambda row: [x.replace(" ", "") for x in row])
        movies['cast'] = movies['cast'].apply(lambda row: [x.replace(" ", "") for x in row])
        movies['crew'] = movies['crew'].apply(lambda row: [x.replace(" ", "") for x in row])
        movies['tags'] = movies['overview'] + movies['keywords'] + movies['cast'] + movies['crew']
        return movies[['id', 'title', 'tags']]
    
    def preprocess(self,df):

        df['tags'] = df['tags'].apply(lambda row: ",".join(row))
        df['tags'] = df['tags'].apply(lambda x:x.lower())
        df['tags'] = df['tags'].apply(lambda x: x.replace(',', ' '))
        df['tags'] = df['tags'].apply(self.stem)
        vectors = self.cv.fit_transform(df['tags']).toarray()
        similarity = cosine_similarity(vectors)
        return df, pd.DataFrame(similarity)
    
    def main(self):

        movie_path = 'archive/tmdb_5000_movies.csv'
        credit_path = 'archive/tmdb_5000_credits.csv'
        df = self.preparation(movie_path, credit_path)
        df, similarity_df = self.preprocess(df)
        df.to_pickle('movies.pkl')
        similarity_df.to_pickle('similarity.pkl')

if __name__ == '__main__':
    mrs = MRS()
    mrs.main()

    