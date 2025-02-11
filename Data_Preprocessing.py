import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""This file will load all datasets pre process them and store them with the same name but pickle format"""
# Importing datasets
Dataframe_news_articles = pd.read_csv('news_articles.csv')
Dataframe_train = pd.read_csv('train_users.csv')
Dataframe_test = pd.read_csv('test_users.csv')


# Checking for missing values 


