#Imports
import streamlit as st
from pyairtable import Table
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import openai

# Page Config
st.set_page_config(page_title="Cocktail Recommendation", layout="wide")

#APIs
api = st.secrets["api"]
base_id = st.secrets["base_id"]
openai.api_key =st.secrets["openai_api"]

#OPENAI Prompt
prompt = str("""Here are the cocktail names and their recipes:\n
Ingredients: 1.5 oz Mezcal, 1 oz Hibiscus Simple Syrup*, .5 oz Lime Juice, top Soda Water 
Preparation: *Hibiscus Simple Syrup  a cup of dried hibiscus steeping for 30-40 min
Name: Flor de Amaras	

Ingredients:	2 oz Junipero Gin, .75 oz House-made Cranberry Syrup*, .5 oz Lemon Juice, .5 oz Cranberry Juice, .25 oz Lillet Blanc, 4 dash Forgery Earth Day Bitters, mist Laphroaig	
Preparation: *House-made Cranberry syrup: \n-- 2 cups Fresh Cranberries\n-- 1 cup Sugar\n-- 1 cup Water\n-- 2 Bay Leaves\n-- .25 cup Pink Peppercorns\n-- Half Serrano Chile\n-- 4 Sprigs Fresh Rosemary\n\nAdd all ingredients to a pot and heat thoroughly. Simmer on low until cranberries cook down for 25 minutes. Strain and let cool.
Name: The Happy Place

Ingredients: 1500 ml BarSol Selecto Italia Pisco, 750 ml Lemon Juice, 750 ml Pineapple Gomme Syrup*, .5 oz Fee Bros Lemon Bitters, 1 float Old Vine Zin	*Pineapple Gomme: 
Preparation: Mix equal parts (1.5 cups) gum arabic with water over high heat until it all mixes and then let cool for a bit. Then you're gonna make a sugar syrup with 2 parts sugar, 1 part water (4 cups water, 2 cups white granulated sugar) in the same manner over high heat until it mixes, and then add the gum syrup to the mix until everything dissolves and what you're left with is a thick gummy syrup that resembles a whole lot of baby batter. Then cut up 1.5 cups of pineapple chunks and add them to the punch, mix in
Name:Bon Voyage Pisco Punch

Ingredients: 1.5 oz BarSol Primero Quebranta Pisco, .75 oz Dry Vermouth, .5 oz St. Germain, .25 oz Pineapple Syrup*, 1 tbsp Vieux Pontarlier Absinthe Francaise Superieure	
Preparation: *Pineapple Syrup Equal parts pineapple blended with water and sugar and strained
Name: 	Still Life of a Pineapple	

Ingredients: 1.25 oz Luxardo Maraschino Liqueur, 4 drops Acid phosphate, 2 oz BarSol Primero Quebranta Pisco, .75 oz Luxardo Amaro Abano, .25 oz Luxardo Fernet, 3 dashes Scrappy's Aromatic Bitters
Preparation: 1st glass ingredients: Luxardo Maraschino, Acid Phosphate in 2nd glass ingredients: BarSol Quebranta Pisco, Luxardo Amaro Abano, Luxardo Fernet, Scrappy's Aromatic Bitters
Name: The Bittered Valley	

Create an original creative name for the following cocktail:

""")

#Cached functions
@st.cache
### Initiating the open ai gpt-3
def get_response():
    return openai.Completion.create(
    model="text-davinci-002",
    prompt = str(prompt+ "\n Ingredients:\n"+ingredients1+"preparation:\n"+preparation1+"Name: "),
    temperature=0.9,
    max_tokens=5,
    top_p=1,
    frequency_penalty=1.5,
    presence_penalty=1.5
    )
###Pulling up date from airtable
@st.cache
def get_data():
    at = Table(api, base_id, 'Cocktails')
    data = at.all()
    return data
###Creating a DataFrame
@st.cache
def to_df(data):
    airtable_rows = [] 
    for record in data:
        airtable_rows.append(record['fields'])
    return pd.DataFrame(airtable_rows)

#Title
title = "Cocktail Recommendation Engine"
st.title(title)

#Initialization
with st.spinner('Fetching Data..'):
    df = to_df(get_data())
    df = df.set_index(['Field 1'])
    df = df.sort_index()
    flavors = ['Sweet', 'Sour', 'Bitter', 'Salty', 'Astringent','Liqueur']
    alcohol_types = ['Absinthe', 'Brandy', 'Champagne', 'Gin', 'Mezcal', 'Pisco', 'Rum', 'Sambuca', 'Tequilla', 'Vodka', 'Whiskey', 'Wine','Scotch','With Liqueur']


    
###Main screen
st.write("This is the place where you can customize what you want to drink to based on genre and several key cocktail features. Try playing around with different settings and try cocktails recommended by my system!")
st.markdown("##")
with st.container():
    col1, col2 = st.columns(2)
    with col2:
        st.markdown("***Choose your Flavor:***")
        flavor = st.multiselect(
            "",
            flavors)
    with col1:
        st.markdown("***Choose features to include:***")
        alcohol_type = st.multiselect(
            'Select the alcohols you enjoy',
            alcohol_types
        )
    features = pd.DataFrame()
    for word in alcohol_types:
        if word in alcohol_type:
            features.loc[0,word] = 1
        else:
            features.loc[0,word] = 0
    for word in flavors:
        if word in flavor:
            features.loc[0,word] = 1
        else:
            features.loc[0,word] = 0 


#Algorithm
if st.checkbox('RUN'):  
    with st.spinner('Building an Algorithm, beep bop..'):
        train_df = df.loc[:,'Sweet':'Pisco']
        pca = PCA(n_components = 12)
        reduced_df = pca.fit_transform(train_df)
        Kmeans = KMeans(n_clusters = 49)
        Kmeans.fit(reduced_df)
        mydict = {i: np.where(Kmeans.labels_ == i)[0] for i in range(Kmeans.n_clusters)} # !! Get the indices of the points for each corresponding cluster

        ## Assign the clusters to cocktails
        df['Cluster'] = 0
        for row in df.index:
            for key, items in mydict.items():
                if row in items:
                    df['Cluster'][row] = key
        #Train test split 
        X = df.loc[:,'Sweet':'Pisco']
        y = df.loc[:,'Cluster']
        X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.75, random_state=42)

        #Training the classifier
        mlpc = MLPClassifier(activation= 'identity', hidden_layer_sizes= 100, learning_rate= 'adaptive', solver= 'lbfgs')
        mlpc.fit(X_train,y_train)
        df = df.append(features, ignore_index=True)
        predicted_cluster = mlpc.predict(df.loc[df.index[-1]:,'Sweet':'Pisco'])

        #Results returned
        col3 = st.columns(2)
        st.header('Cocktails you might enjoy:')
        st.write("#"*150)
        for index, i in df[df['Cluster']==predicted_cluster[0]].iterrows():
            if str(i['Cocktail Name']) == '' or pd.isnull(i['Cocktail Name']):
                st.subheader('Cocktail Name:')
                response = get_response()
                st.write(response.choices[0].text)
            else:
                st.header('Cocktail Name:')
                st.write(i['Cocktail Name'])
            st.subheader('Ingredients: ')
            st.text(i['Ingredients'])
            st.subheader('Preparation:')
            st.text(i['Preparation'])
            st.write("#"*150)
                

