import os
import subprocess
import streamlit as st
import geopandas as gpd
import requests
from io import BytesIO
import zipfile
import openai  # Assure-toi d'importer openai ici

# Fonction d'installation des packages manquants
def install(package):
    subprocess.check_call(["pip", "install", package])

# Installation automatique des dépendances si nécessaire
libraries = ["geopandas", "requests", "openai", "langchain==0.0.208", "annoy", "tiktoken", "fiona"]
for lib in libraries:
    try:
        __import__(lib.split('==')[0])
    except ImportError:
        install(lib)

# Importations après vérification des dépendances
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Annoy

# Définition de la clé API OpenAI (en production, cela devrait venir des secrets)
openai_api_key = "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"
openai.api_key = openai_api_key  # Cette ligne est correcte après avoir importé openai

# URLs des fichiers Shapefile stockés sur Google Drive
commune_zip_url = "https://drive.google.com/uc?export=download&id=1chzyg5TWgugKcr3Mn9LHl-S6cka_uO4J"
departement_zip_url = "https://drive.google.com/uc?export=download&id=1m1sxx9nQe8A1JiYygtLoUXKiCOyJJMj5"

# Fonction pour télécharger et décompresser un fichier ZIP contenant des Shapefiles
def download_and_extract_shapefile(url, extract_to="shapefiles"):
    response = requests.get(url)
    zip_file = BytesIO(response.content)
    
    # Décompresser le fichier ZIP
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(extract_to)
    return extract_to

# Télécharger et décompresser les fichiers ZIP depuis Google Drive
commune_dir = download_and_extract_shapefile(commune_zip_url)
departement_dir = download_and_extract_shapefile(departement_zip_url)

# Lire les fichiers Shapefile avec Geopandas (en supposant que le fichier .shp se trouve dans le répertoire extrait)
gdf_commune = gpd.read_file(f"{commune_dir}/communes_2024T2_v3.shp")
gdf_departement = gpd.read_file(f"{departement_dir}/dpt_2024T2_v2.shp")

# Calcul des taux de couverture FTTH pour les communes
gdf_commune['Locaux'] = gdf_commune['Locaux'].astype(int)
gdf_commune['taux_exact_couv'] = (gdf_commune['ftth'] / gdf_commune['Locaux']) * 100
gdf_commune['taux_exact_couv'] = gdf_commune['taux_exact_couv'].round(2)

# Calcul des taux de couverture FTTH pour les départements
gdf_departement['Locaux'] = gdf_departement['Locaux'].astype(int)
gdf_departement['taux_exact_couv'] = (gdf_departement['ftth'] / gdf_departement['Locaux']) * 100
gdf_departement['taux_exact_couv'] = gdf_departement['taux_exact_couv'].round(2)

# Préparation des données pour le modèle LLM (Langchain)
def prepare_commune_documents(df):
    documents = []
    for _, row in df.iterrows():
        content = (f"Commune: {row['NOM_COM']}, Population: {row['POPULATION']}, "
                   f"FTTH: {row['ftth']}, Locaux: {row['Locaux']}, "
                   f"Couverture: {row['taux_exact_couv']}%")
        documents.append({"page_content": content, "metadata": {"commune": row['NOM_COM']}})
    return documents

commune_documents = prepare_commune_documents(gdf_commune)

# Utilisation du modèle OpenAI pour les embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Création des embeddings pour les communes
commune_embeddings = embedding_model.embed_documents([doc["page_content"] for doc in commune_documents])

# Création de l'index Annoy pour les recherches rapides
dimension_commune = len(commune_embeddings[0])
commune_index = Annoy(dimension_commune, "angular")
for i, embedding in enumerate(commune_embeddings):
    commune_index.add_item(i, embedding)
commune_index.build(10)

# Interface utilisateur Streamlit
st.title("Assistant IA - Taux de couverture FTTH")

# Champ de texte pour que l'utilisateur pose des questions
user_query = st.text_input("Posez votre question sur la couverture FTTH :")

# Lorsqu'on clique sur le bouton
if st.button("Générer une réponse"):
    if user_query:
        # Embedding de la requête utilisateur
        query_embedding = embedding_model.embed_documents([user_query])[0]

        # Recherche des communes similaires via Annoy
        indices = commune_index.get_nns_by_vector(query_embedding, 5)
        similar_documents = [commune_documents[i]["page_content"] for i in indices]

        # Affichage des documents similaires
        st.write("Documents pertinents :")
        for doc in similar_documents:
            st.write(f"- {doc}")
    else:
        st.write("Veuillez entrer une question.")
