import streamlit as st
import geopandas as gpd
import requests
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Annoy
import openai
import os

# Clé API OpenAI (à configurer dans les secrets Streamlit Cloud)
openai_api_key = os.getenv("OPENAI_API_KEY")

# URLs des fichiers Shapefile stockés sur Google Cloud Storage
commune_shp_url = "https://storage.googleapis.com/ton-bucket/communes_2024T2_v3.shp"
departement_shp_url = "https://storage.googleapis.com/ton-bucket/dpt_2024T2_v2.shp"

# Fonction pour télécharger et lire le fichier Shapefile
def download_shapefile(url):
    response = requests.get(url)
    return BytesIO(response.content)

# Télécharger les Shapefiles depuis Google Cloud Storage
commune_file = download_shapefile(commune_shp_url)
departement_file = download_shapefile(departement_shp_url)

# Lire les fichiers Shapefile avec Geopandas
gdf_commune = gpd.read_file(commune_file)
gdf_departement = gpd.read_file(departement_file)

# Calcul des taux de couverture FTTH pour les communes
gdf_commune['Locaux'] = gdf_commune['Locaux'].astype(int)
gdf_commune['taux_exact_couv'] = (gdf_commune['ftth'] / gdf_commune['Locaux']) * 100
gdf_commune['taux_exact_couv'] = gdf_commune['taux_exact_couv'].round(2)

# Calcul des taux de couverture FTTH pour les départements
gdf_departement['Locaux'] = gdf_departement['Locaux'].astype(int)
gdf_departement['taux_exact_couv'] = (gdf_departement['ftth'] / gdf_departement['Locaux']) * 100
gdf_departement['taux_exact_couv'] = gdf_departement['taux_exact_couv'].round(2)

# Préparation des données pour le LLM
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
