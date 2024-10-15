import streamlit as st
import geopandas as gpd
from langchain_openai import OpenAIEmbeddings
from annoy import AnnoyIndex
import os

# Exemple de chargement des fichiers .shp depuis des URLs (à ajuster avec les bons liens)
commune_shp_url = "https://ton-lien-vers-le-fichier-communes.shp"
departement_shp_url = "https://ton-lien-vers-le-fichier-departement.shp"

# Charger les fichiers Shapefile des communes et des départements
gdf_commune = gpd.read_file(commune_shp_url, encoding='utf-8')
gdf_departement = gpd.read_file(departement_shp_url, encoding='utf-8')

# Calcul des taux de couverture FTTH pour les communes
gdf_commune['Locaux'] = gdf_commune['Locaux'].astype(int)
gdf_commune['taux_exact_couv'] = (gdf_commune['ftth'] / gdf_commune['Locaux']) * 100
gdf_commune['taux_exact_couv'] = gdf_commune['taux_exact_couv'].round(2)

# Calcul des taux de couverture FTTH pour les départements
gdf_departement['Locaux'] = gdf_departement['Locaux'].astype(int)
gdf_departement['taux_exact_couv'] = (gdf_departement['ftth'] / gdf_departement['Locaux']) * 100
gdf_departement['taux_exact_couv'] = gdf_departement['taux_exact_couv'].round(2)

# Initialisation du modèle d'OpenAI pour les embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")  # Utiliser une variable d'environnement pour la clé OpenAI
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Préparer les documents des communes pour l'embedding
def prepare_commune_documents(df):
    documents = []
    for _, row in df.iterrows():
        content = f"Commune: {row['NOM_COM']}, Population: {row['POPULATION']}, FTTH: {row['ftth']}%, Locaux: {row['Locaux']}, Couverture: {row['taux_exact_couv']}%"
        documents.append({"page_content": content, "metadata": {"commune": row['NOM_COM']}})
    return documents

commune_documents = prepare_commune_documents(gdf_commune)
commune_embeddings = embedding_model.embed_documents([doc["page_content"] for doc in commune_documents])

# Création de l'index Annoy pour les communes
dimension_commune = len(commune_embeddings[0])
commune_index = AnnoyIndex(dimension_commune, 'angular')
for i, embedding in enumerate(commune_embeddings):
    commune_index.add_item(i, embedding)
commune_index.build(100)

# Interface utilisateur
st.title("Assistant IA - Taux de couverture FTTH")

# Entrée de la question utilisateur
user_query = st.text_input("Posez votre question sur la couverture FTTH :")

# Si l'utilisateur clique sur le bouton
if st.button("Générer une réponse"):
    if user_query:
        # Embedder la requête utilisateur
        query_embedding = embedding_model.embed_documents([user_query])[0]

        # Rechercher les documents similaires avec Annoy
        indices = commune_index.get_nns_by_vector(query_embedding, 5)
        similar_documents = [commune_documents[i]["page_content"] for i in indices]

        # Afficher les documents pertinents
        st.write("Documents pertinents :")
        for doc in similar_documents:
            st.write(f"- {doc}")
    else:
        st.write("Veuillez entrer une question.")
