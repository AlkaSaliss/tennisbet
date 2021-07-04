import os
import pathlib
import pickle
import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model
import plotly.express as px


# Constant vars
MODEL_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), "artifacts/model/best_lgbm")
PREPROCESSING_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), "artifacts/model/preprocessing_pipeline.pklz")
CATEGORIES_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), "artifacts/model/dict_categories.pklz")
st.set_page_config(page_title='TennisBet', page_icon="🎾🏆", layout='wide', initial_sidebar_state='auto')


def load_artifacts():
    """Charger les artifacts (modèle, dictionnaire de categories et pipeline de preprocessing) """
    model = load_model(MODEL_PATH)
    
    with open(PREPROCESSING_PATH, "rb") as f:
        preprocessing_pipeline = pickle.load(f)
    
    with open(CATEGORIES_PATH, "rb") as f:
        dict_categories = pickle.load(f)
    
    return model, preprocessing_pipeline, dict_categories


def predict(model, df):
    preds = predict_model(model, data=df)
    prob1 = round(100*preds.Score.item(), 2)
    prob2 = 100 - prob1
    winner = "Joueur 1" if prob1 > prob2 else "Joueur 2"
    preds = pd.DataFrame({"Joueurs": ["Joueur 1", "Joueur 2"], "Chance": [prob1, prob2]})
    st.info(f"Il semblerait que le **{winner}** ait plus de chance de gagner 🎾💪...")
    fig = px.bar(preds, x='Joueurs', y='Chance', text="Chance")
    fig.update_layout(title_text="Chance de gagner le prochain match pour les deux joueurs", title_x=0.5)
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig)


# Charger le modèle
model, preprocessing_pipeline, dict_categories = load_artifacts()


def main():
    print("#######################################", pd.__version__)
    
    st.header("Prédiction du vainqueur d'un match de tennis ​🎾​🏆​")
    st.write("""Cette application permet de prédire les chances de victoire pour chacun des deux adversaires d'un match de tennis.""")
    st.sidebar.header("Entrez les caractéristiques des joueurs et du match")
    st.sidebar.write("""Utiliser le formulaire ci-dessous pour renseigner les caractéristiques des deux joueurs ainsi que ceux du match. Les prédictions sont mises à jour automatiquement.""")
    
    
    # pour choisir les différents caractéristiques des joueurs et du match
    age1 = st.sidebar.slider("Age joueur 1", min_value=10, max_value=80, value=25, step=1)
    age2 = st.sidebar.slider("Age joueur 2", min_value=10, max_value=80, value=25, step=1)
    
    height1 = st.sidebar.slider("Taille joueur 1", min_value=50, max_value=250, value=180, step=1)
    height2 = st.sidebar.slider("Taille joueur 2", min_value=50, max_value=250, value=180, step=1)
    
    rank1 = st.sidebar.number_input("Rang joueur 1", min_value=0, step=1, value=0)
    rank2 = st.sidebar.number_input("Rang joueur 2", min_value=0, step=1, value=0)

    points1 = st.sidebar.number_input("Points joueur 1", min_value=0., step=1., value=0.)
    points2 = st.sidebar.number_input("Points joueur 2", min_value=0., step=1., value=0.)
    
    hand1 = st.sidebar.selectbox("Main joueur 1", options=dict_categories["player1_hand"])
    hand2 = st.sidebar.selectbox("Main joueur 2", options=dict_categories["player2_hand"], index=1)

    country1 = st.sidebar.selectbox("Pays du joueur 1", options=dict_categories["player1_ioc"])
    country2 = st.sidebar.selectbox("Pays du joueur 2", options=dict_categories["player2_ioc"], index=1)
    
    match_num = st.sidebar.number_input("Numéro du match dans le tournoi", min_value=0, step=1, value=0)
    round_ = st.sidebar.selectbox("Round", options=dict_categories["round"])
    surface = st.sidebar.selectbox("Type de surface", options=dict_categories["surface"])
    tourney_level = st.sidebar.selectbox("Niveau du tournoi", options=dict_categories["tourney_level"])
    
    df = pd.DataFrame([[age1, hand1, height1, country1, rank1, points1, 
                       age2, hand2, height2, country2, rank2, points2,
                       match_num, round_, surface, tourney_level, None]], 
                     columns=["player1_age", "player1_hand", "player1_ht", "player1_ioc", 
                              "player1_rank", "player1_rank_points", "player2_age",
                              "player2_hand", "player2_ht", "player2_ioc", "player2_rank",
                              "player2_rank_points", "match_num", "round", "surface",
                              "tourney_level", "is_player1_winner"])
    
    # Imputer les valeurs manquantes
    df = preprocessing_pipeline.transform(df)
    
    # Récuperer la liste des colonnes numériques et celle des colonnes catégorielles
    list_numeric_cols = ["player1_age", "player1_ht", "player1_rank", "player1_rank_points",
                         "player2_age", "player2_ht",  "player2_rank", "player2_rank_points",
                         "match_num"]
    list_categ_cols = ["player1_hand", "player1_ioc", "player2_hand", "player2_ioc",
                       "round", "surface", "tourney_level"]
    target_col = "is_player1_winner"  # colonne à prédire
    
    df = pd.DataFrame(df, columns=list_numeric_cols+list_categ_cols+[target_col])
    
    st.write("### Récapitulatif des valeurs entrées")
    st.write(df)
    # Prédire
    predict(model, df)
    


if __name__ == "__main__":
    main()
