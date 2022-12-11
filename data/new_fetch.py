import requests
import csv
import json
import pandas as pd
import re

#loading books.csv
book_cat = pd.read_csv("/Users/liviagimenes/Documents/CS/CS1470/longformer/data/books_and_genres.csv")


print(book_cat)

book_cat = book_cat.dropna()

book_cat["genres"] = book_cat["genres"].str.strip("{}")
book_cat["genres"]  = book_cat["genres"].str.split(",")   
book_cat["genres"]  = book_cat["genres"].apply(lambda x: [s.strip("'") for s in x])

book_cat = book_cat[book_cat["genres"].apply(lambda x: len(x) == 1)]

def less_than_thousand(word_list):
    #remove the non word characters
    word_list = [re.sub(r'\W', '', s) for s in word_list]
    if len(word_list) <= 1000:
        return word_list
    else:
        word_list = word_list[:1000]
        return word_list



book_cat['text'] = book_cat['text'].map(lambda x: x.split(" "))
book_cat['text'] = book_cat['text'].map(lambda x: less_than_thousand(x))
book_cat['text'] = book_cat['text'].map(lambda x: " ".join(x))
for i in book_cat['text']:
    print(i)
print(book_cat)


#get a list of unique genres 
book_cat['genres'] = book_cat['genres'].map(lambda x: str(x).replace("['", "").replace("']", "").strip("[']"))
genre_list = []
for i in book_cat['genres']:
    if i not in genre_list:
        genre_list.append(i)

def match_encodings(genre):
    if genre in genre_list:
        enconding_idx = genre_list.index(genre)
        return enconding_idx

book_cat['genre_encodings'] = book_cat['genres'].map(lambda x: match_encodings(x))


print(book_cat)

book_cat.to_csv("/Users/liviagimenes/Documents/CS/CS1470/longformer/data/book_one_genre.csv")

