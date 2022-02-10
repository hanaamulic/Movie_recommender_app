import streamlit as st
from numpy import load
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle


st.set_page_config(layout="wide")

API_key = 'd06b517b45b8643d8cdb739e04465106'

# all_movies = pd.read_csv('data/movies_metadata.csv')
# all_movies = all_movies.drop([19730,29503,35587],axis=0)
# all_movies['id'] = all_movies['id'].astype(str).astype(int)

top_movies=pd.read_csv('data/top_movies.csv')

simple_reccomenders = pd.read_csv('data/second_recommender.csv')

genres = pd.read_csv('data/genres.csv')
genres = genres.append({'id':1,'genre':'All'},ignore_index=True)

rating_movies = pd.read_csv('data/rating_movies.csv')


@st.experimental_memo
def get_cos_sim(series):
    
    if(series.name == 'description'):
        tf = TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=0,stop_words='english')
        tfidf_matrix = tf.fit_transform(series.values.astype('U'))
        cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return cos_sim
    else:
        count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        count_matrix = count.fit_transform(series.values.astype('U'))
        cosine_sim_second = cosine_similarity(count_matrix, count_matrix)
        return cosine_sim_second

    
@st.experimental_memo
def get_content_based_movies(title,series):
    
    cos_sim = get_cos_sim(series)
    title_id = pd.Series(simple_reccomenders.reset_index().index, index=simple_reccomenders['title'])
    index = title_id[title]
    if isinstance(index, (np.int64)):
        index = title_id[title]
    elif isinstance(index, (pd.Series)):
        index = title_id[title][0]
    else:
        index = title_id[title]
    sim_scores = list(enumerate(cos_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    content_based_movies = simple_reccomenders.iloc[movie_indices]
    content_based_movies = content_based_movies.sort_values('weighted_rate',ascending=False).reset_index()
    return content_based_movies


def get_predicted_ratings(alg):
    
    if alg == 'svd':
        predict_dict = pickle.load(open('models/predict_dict_user_based.sav', 'rb'))
    else:
        predict_dict = pickle.load(open('models/predict_dict_knnz.sav', 'rb'))
    predicted_ratings = pd.DataFrame(predict_dict.items(), columns=['userId', 'est_score'])
    df = pd.DataFrame(predict_dict.items(), columns=['userId', 'est_score'])
    return df

def get_users_info(userId):
        real_values = rating_movies[rating_movies.userId == userId].copy().sort_values('rating',ascending = False).head(10)

        predicted_ratings = get_predicted_ratings('svd') 
        user_ratings = pd.DataFrame (predicted_ratings.est_score[userId], columns = ['movieId','est_score'])
        estimated_values_svd = user_ratings.merge(rating_movies, how='left', left_on='movieId', right_on = 'movieId').drop_duplicates('movieId', keep='first')

        predicted_ratings = get_predicted_ratings('knnz')
        user_ratings = pd.DataFrame (predicted_ratings.est_score[userId], columns = ['movieId','est_score'])
        estimated_values_knnz = user_ratings.merge(rating_movies, how='left', left_on='movieId', right_on = 'movieId').drop_duplicates('movieId', keep='first')

        return real_values, estimated_values_svd, estimated_values_knnz
    
#-----------------------------------------------------------------------------------------------------------------------------------------------#

st.title("Movie recommenders")



##Top movies
#----------------------------------------------------#
st.header('Simple recommender')
st.write('This recommender is based on weighted ratings. Movies can be divided by genres.')
top_container = st.container()

with top_container:
    with st.form('top_form'):
        col3, col1, col2 = st.columns([.1,.5,1])
        with col3:
            st.write('.\n')
            st.write('Genre:')
        with col1:
            genre = st.selectbox('', (genres.genre), index=20)
        with col2:
            st.write('.\n')
            submit_top = st.form_submit_button('Show')
        st.subheader(f'''Top 10 movies in category: {genre}''')
        temp=[]
        width = [3]*11
        temp[0:10] = st.columns(width)
        if (genre == 'All'):
            for i, row in top_movies.head(10).iterrows():
                with temp[i]:
                    # st.caption(row['title'])
                    # UNNCOMENT FOR REAL APP
                    response = requests.get(f"""https://api.themoviedb.org/3/find/{row['imdb_id']}?api_key={API_key}&language=en-US&external_source=imdb_id""")
                    try:
                        st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                    except:
                        # COMMENT FOR REAL APP
                        st.image(('rsz_movie_default.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                    # COMMENT FOR REAL APP
                    # st.image(('https://image.tmdb.org/t/p/w92' + '/2gvbZMtV1Zsl7FedJa5ysbpBx2G.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
        else:
            top_in_genre = top_movies[top_movies.genres.str.contains(genre)].reset_index()
            for j, row in top_in_genre.head(10).iterrows():
                with temp[j]:
                    # st.caption(row['title'])
                    # UNNCOMENT FOR REAL APP
                    response = requests.get(f"""https://api.themoviedb.org/3/find/{row['imdb_id']}?api_key={API_key}&language=en-US&external_source=imdb_id""")
                    try:
                        st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                    except:
                        # COMMENT FOR REAL APP
                        st.image(('rsz_movie_default.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                    # COMMENT FOR REAL APP
                    # st.image(('https://image.tmdb.org/t/p/w92' + '/2gvbZMtV1Zsl7FedJa5ysbpBx2G.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
            

##Content movies
#----------------------------------------------------------------------------------------------#
st.header('Item based recommender (Summary)')
st.write('This recommender is based on similarities of movie summaries. Provided a movie it will recommend 10 movies with similar storyline')

content_container = st.container()
with content_container:
    with st.form('content_form'):
        col3, col1, col2 = st.columns([.15,.5,1])
        with col3:
            st.write('.\n')
            st.write('Movie you liked:')
        with col1:
            title = st.text_input('', value="",  help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder='e.g. The Godfather', disabled=False)
        with col2:
            st.write('.\n')
            submit_content = st.form_submit_button('Show')  
        
        if title == '':
            title = 'The Godfather'
        
        if (title in simple_reccomenders.title.unique() ) | (simple_reccomenders.title.str.contains(fr"{title}", case=False).any()):
            if title in simple_reccomenders.title.unique():
                title = title
            else:
                title = simple_reccomenders[simple_reccomenders.title.str.contains(rf"{title}",case=False,na=False)]['title'].to_list()[0]
            st.subheader(f'''Because you liked: {title}''')
            temp1=[]
            width = [3]*11
            temp1[0:10] = st.columns(width)
            content_based_movies=get_content_based_movies(title,simple_reccomenders.description)
            for j, row in content_based_movies.head(10).iterrows():
                    with temp1[j]:
                        # UNNCOMENT FOR REAL APP
                        response = requests.get(f"""https://api.themoviedb.org/3/find/{row['imdb_id']}?api_key={API_key}&language=en-US&external_source=imdb_id""")
                        try:
                            st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                        except:
                        # COMMENT FOR REAL APP
                            st.image(('rsz_movie_default.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                        # COMMENT FOR REAL APP
                        # st.image(('https://image.tmdb.org/t/p/w92' + '/2gvbZMtV1Zsl7FedJa5ysbpBx2G.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
        
        else:
            st.write('No such movie in database!')                    
                    
         
        
## Actor director movies
##----------------------------------------------------------------------------------------##
st.header('Item based recommender (cast, crew)')
st.write('This recommender is based on similarities in several categories: Director, leading cast.')   

actor_director_container = st.container()
with actor_director_container:
    with st.form('actor_dir_form'):
        col3, col1, col2 = st.columns([.15,.5,1])
        with col3:
            st.write('.\n')
            st.write('Movie you liked:')
        with col1:
            title = st.text_input('', value="",  help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder='e.g. The Godfather', disabled=False)
        with col2:
            st.write('.\n')
            submit_content = st.form_submit_button('Show')  
        
        if title == '':
            title = 'The Godfather'
        
        
        if (title in simple_reccomenders.title.unique() ) | (simple_reccomenders.title.str.contains(fr"{title}", case=False).any()):
            if title in simple_reccomenders.title.unique():
                title = title
            else:
                title = simple_reccomenders[simple_reccomenders.title.str.contains(rf"{title}",case=False,na=False)]['title'].to_list()[0]
            st.subheader(f'''Because you liked: {title}''')
            temp1=[]
            width = [3]*11
            temp1[0:10] = st.columns(width)
            content_based_movies=get_content_based_movies(title,simple_reccomenders.soup)
            for j, row in content_based_movies.head(10).iterrows():
                    with temp1[j]:
                        # UNNCOMENT FOR REAL APP
                        response = requests.get(f"""https://api.themoviedb.org/3/find/{row['imdb_id']}?api_key={API_key}&language=en-US&external_source=imdb_id""")
                        try:
                            st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                        except:
                        # COMMENT FOR REAL APP
                            st.image(('rsz_movie_default.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                        # COMMENT FOR REAL APP
                        # st.image(('https://image.tmdb.org/t/p/w92' + '/2gvbZMtV1Zsl7FedJa5ysbpBx2G.jpg'), caption=row['title'], width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
        
        else:
            st.write('No such movie in database!')                    
                    

## Userbased movies
##----------------------------------------------------------------------------------------#

st.header('User based recommender')
st.write('This recommender is based on similarities between users.')

user_based_container = st.container()
with user_based_container:
    with st.form('user_based_form'):
        
        col3, col1, col2 = st.columns([.15,.5,1])
        with col3:
            st.write('.\n')
            st.write('Choose user:')
        with col1:
            userId = st.number_input('', value=5, min_value=0, step=1, max_value=999)
            userId = int(userId)
        with col2:
            st.write('.\n')
            submit_user_based = st.form_submit_button('Show')
         
        
        if (userId >= 1 & userId <= 671):
            real_values, estimated_values_svd, estimated_values_knnz = get_users_info(userId)
            
            st.subheader(f'''Real ratings for user {userId}''')
            temp2=[]
            width = [3]*11
            temp2[0:10] = st.columns(width)
            
            for k, row in real_values.reset_index().head(10).iterrows():
                    with temp2[k]:
                        caption = f"{row['title']} ({row['rating']})"
                        # UNNCOMENT FOR REAL APP
                        response = requests.get(f"""https://api.themoviedb.org/3/find/{row['imdb_id']}?api_key={API_key}&language=en-US&external_source=imdb_id""")
                        try:
                            st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=caption, width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                        except:
                        # COMMENT FOR REAL APP
                            st.image(('rsz_movie_default.jpg'), caption=caption, width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto") 
                        # COMMENT FOR REAL APP
                        # st.image(('https://image.tmdb.org/t/p/w92' + '/2gvbZMtV1Zsl7FedJa5ysbpBx2G.jpg'), caption=caption, width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                        
            st.subheader(f'''Estimated ratings based on SVD algorithm for user {userId}''')
            temp3=[]
            width = [3]*11
            temp3[0:10] = st.columns(width)
            for l, row in estimated_values_svd.reset_index().head(10).iterrows():
                    with temp3[l]:
                        caption = f"{row['title']} ({row['est_score']:.2f})"
                        # UNNCOMENT FOR REAL APP
                        response = requests.get(f"""https://api.themoviedb.org/3/find/{row['imdb_id']}?api_key={API_key}&language=en-US&external_source=imdb_id""")
                        try:
                            st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=caption, width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                        except:
                        # COMMENT FOR REAL APP
                            st.image(('rsz_movie_default.jpg'), caption=caption, width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")  
                        
                        
                        
            st.subheader(f'''Estimated ratings based on KNNZ algorithm for user {userId}''')
            temp4=[]
            width = [3]*11
            temp4[0:10] = st.columns(width)
            for m, row in estimated_values_knnz.reset_index().iterrows():
                    with temp4[m]:
                        caption = f"{row['title']} ({row['est_score']:.2f})"
                        # UNNCOMENT FOR REAL APP
                        
                        response = requests.get(f"""https://api.themoviedb.org/3/find/{row['imdb_id']}?api_key={API_key}&language=en-US&external_source=imdb_id""")
                        try:
                            st.image(('https://image.tmdb.org/t/p/w92' + response.json()['movie_results'][0]['poster_path']), caption=caption, width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
                        except:
                        # COMMENT FOR REAL APP
                            st.image(('rsz_movie_default.jpg'), caption=caption, width=None, use_column_width='always', clamp=False, channels="RGB", output_format="auto")
        else:
            st.write('Insert existing user')     
