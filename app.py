
import streamlit as st
import numpy as np
import joblib

from pathlib import Path
from PIL import Image, ImageOps
from features import extract_all_features

# Configuration de l'application
st.set_page_config(
    page_title="CBIR Image Classifier",
    page_icon="",
    layout="centered"
)

# Chemin du projet
BASE_DIR = Path(__file__).resolve().parent

# Charger les fichiers du modèle
@st.cache_resource
def charger_modele():
    modele = joblib.load(
        BASE_DIR /"svm_sigmoid.joblib"
    )

    scaler = joblib.load(
        BASE_DIR /"scaler.joblib"
    )

    return modele, scaler

modele, scaler = charger_modele()

# Interface
st.title("Classification d'images avec CBIR")
st.write("Application de classification supervisée : Car / Ship")

st.info(
    "Le modèle reconnaît uniquement les classes "
    "Car (voiture) et Ship (bateau)."
)

st.subheader("Importer une image")

uploaded_file = st.file_uploader(
    "Choisissez une image JPG ou PNG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Lire l'image importée
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    st.image(
        image,
        caption="Image importée",
        width="stretch"
    )

    if st.button("Classifier l'image"):

        with st.spinner("Analyse de l'image..."):

            # Image PIL vers NumPy RGB
            img_array = np.array(image)

            # Extraction des caractéristiques CBIR
            features = extract_all_features(img_array)

            # Vérification
            if features.shape[0] != 539:
                st.error("Nombre de caractéristiques incorrect.")
                st.stop()

            if not np.isfinite(features).all():
                st.error("Caractéristiques non valides.")
                st.stop()

            # Mise en forme : (1, 539)
            features = features.reshape(1, -1)

            # Standardisation
            features_scaled = scaler.transform(features)

            # Classification
            prediction = modele.predict(features_scaled)[0]

            # Correspondance classe / nom
            classes = {
                0: "Car (Voiture)",
                1: "Ship (Bateau)"
            }

            st.success(
                f"Classe prédite : {classes[int(prediction)]}"
            )

            st.write("Modèle utilisé : SVM sigmoïde")
            st.write("Descripteurs : RGB, HSV, GLCM, LBP, Hu")
            st.write("Nombre de caractéristiques : 539")
