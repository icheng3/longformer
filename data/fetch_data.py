import requests
import csv
import json
import pandas as pd

#loading books.csv
book_cat = pd.read_csv("/Users/liviagimenes/Documents/CS/CS1470/longformer/data/books_and_genres.csv")


print(book_cat)




#loading test and train from booksum 
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")



train_data = train_data.drop(["summary_name","content","summary",'bid',"is_aggregate","source","chapter_path","summary_path","summary_url","summary_analysis","summary_length","analysis_length"],axis=1)
test_data = test_data.drop(["summary_name","content","summary",'bid',"is_aggregate","source","chapter_path","summary_path","summary_url","summary_analysis","summary_length","analysis_length"],axis=1)








