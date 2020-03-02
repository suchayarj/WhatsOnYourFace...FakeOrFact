import pandas as pd

#import reviews from Sephora from 2 sources
df1 = pd.read_csv('data/sephora_review_ordinary.csv')
df2 = pd.read_csv('data/sephora_review.csv')

#combine datasets
def combine_data(df1, df2):
    df1_test =df1[['r_review','r_star']]
    df2_test = df2[['review_text', 'rating']]
    df1_test.rename(columns = {'r_review':'Review', 'r_star': 'Rating'}, inplace = True)
    df2_test.rename(columns = {'review_text':'Review', 'rating': 'Rating'}, inplace = True)
    sephora = pd.concat([df1_test, df2_test])
    sephora.to_csv('data/sepho_review_rating.csv')
    return sephora

