
"""
Created on Thu Sep  1 20:12:15 2016
Purpose: Start using Machine Learning skills
@author: joshWeston
"""

import os
import pandas as pd
import sqlite3

#Move to Repository Directory
os.chdir("/Users/joshWeston/Git Repositories/Kaggle-Iris/")

#List all available tables
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#cursor.fetchall()

#Connect to database
con = sqlite3.connect('database.sqlite')

#Create pandas dataframe
df = pd.read_sql('SELECT * FROM IRIS', con)

#Calculate entropy (information gain) of specific features
#entropy of 0 is no information gain
#entropy of 1 is 100% information gain

#We need to partiton the dataset on features to see if we are achieving information gain

#1) Calculate full group entropy
# This is fairly simply with binary classification
#(24/49.0)*(math.log((24/49.0), 2)) + (25/49.0)*(math.log((25/49.0), 2))

bySpecies = df.groupby('Species')
bySpecies.describe()

#50 of each kind. So group entropy is:


#Looks like a decent clustering problem.
#Petal length and petal width appear ot provide the most information gain, but let's
#run each feature against entropy calculation to be sure. Don't forget to use hold-out
#and cross-validation. In Tableau, can we plot the original clustering against our
#clustering to see how they overlap?