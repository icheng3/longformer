import pandas as pd
book_cat = pd.read_csv("/Users/irischeng/Documents/CS1470/longformer/longformer/book_one_genre.csv")



df2 = book_cat.drop(labels=['id'])
df2.to_csv("./new_book_genre.csv")