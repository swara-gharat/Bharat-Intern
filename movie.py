import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

movie_ratings=pd.read_csv('D:/SWARA/internship 2/ml-25m/ratings.csv')

movie_titles=pd.read_csv('D:/SWARA/internship 2/ml-25m/movies.csv')

print(movie_ratings)


print(movie_titles)

movie_titles.drop('genres',inplace=True,axis=1)
movie_ratings.drop('timestamp',inplace=True,axis=1)

#MERGING THE TWO DATAFRAMES ON THE BASIS OF COMMON COLUMN 'MOVIEID'

df=movie_ratings.merge(movie_titles,on='movieId')
print(df)

#CREATING A 'RATINGS' DATAFRAME CONSISTING OF MOVIE TITLES AND THIER AVERAGE RATING GIVEN BY PEOPLE
ratings=pd.DataFrame(df.groupby('title')['rating'].mean())

#RENAMING 'RATINGS' TO 'AVERAGE RATINGS'
ratings.rename(columns={'rating':'average rating'},inplace=True)

#CREATING A COLUMN CALLED 'NUM OF RATINGS' DENOTING HOW MANY PEOPLE GAVE RATINGS TO THE MOVIE
ratings['num of ratings']=pd.DataFrame(df.groupby('title')['rating'].count())

print(ratings)

#MERGING THE MOVIE TITLE, MOVIE RATINGS DATAFRAME WITH THE RATINGS DATAFRAME ON COMMON COLUMN 'TITLE'
df_with_average_rating=df.merge(ratings,on='title')

print(df_with_average_rating)

#GRABBING ONLY THOSE MOVIES WITH AVERAGE RATING GREATER THAN 4 STARS FOR SHRINKING THE ENORMOUS DATASET
movies_above_4_stars=df_with_average_rating[df_with_average_rating['average rating']>=4]

print(movies_above_4_stars)


#CREATING A MATRIX USING PIVOT TABLE FEATURE OF PANDAS, WITH USER ID AS INDEX, MOVIE TITLE AS COLUMN AND RATINGS AS VALUES
moviemat = movies_above_4_stars.pivot_table(index='userId',columns='title',values='rating')
print(moviemat)

#GRABBING THE MOVIES WITH MOST NUMBER OF RATINGS (THE MOST POPULAR ONES) AND AVEARGE RATING GREATER THAN 4
ratings_new=ratings[ratings['average rating']>=4]
movies=ratings_new.sort_values('num of ratings',ascending=False)
print(movies)

#CHECKING THE TOP 60 MOVIES WITH MOST RATINGS AND HIGHEST AVERAGE RATINGS
movies.head(10)

#TAKING MOVIE NAME AS INPUT (IN THIS CASE-STAR WARS:EPISODE 4)
movie=input("Please select a movie you liked:")

desired_movie_user_ratings=moviemat[movie]

#USING CORRWITH() TO FIND SIMILAR MOVIES AS THE ONE SELECTED
similar_to_desired_movie=moviemat.corrwith(desired_movie_user_ratings)

#STORING THE CORRELATION DATA IN A DATAFRAME AND DROPPING THE NULL VALUES
corr_desired_movie = pd.DataFrame(similar_to_desired_movie,columns=['Correlation'])
corr_desired_movie.dropna(inplace=True)

#MERGING THE RATINGS_NEW DATAFRAME WITH CORRELATION DATAFRAME FOR SEEING THE TITLES
corr_desired_movie = corr_desired_movie.merge(ratings_new,on='title')

print(corr_desired_movie[corr_desired_movie['num of ratings']>10000].sort_values('Correlation',ascending=False).head(10))



