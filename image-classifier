from PIL import Image, ImageTk
from tkinter import filedialog
from io import BytesIO
from google_images_search import GoogleImagesSearch
import numpy as np
import requests
import tensorflow as tf
import tkinter as tk

# Charger le modèle pré-entrainé MobileNetV2
model = tf.keras.applications.MobileNetV2(weights="imagenet")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Fonction pour obtenir les URLs des images à partir de Google Images Search
def get_image_urls(api_key, cx, query):
    gis = GoogleImagesSearch(api_key, cx)

    # Paramètres de la recherche d'images
    search_params = {
        'q': query,
        'num': 5  # Nombre d'images à récupérer
    }
    gis.search(search_params, width=500, height=400)  # Taille spécifiée des images

    # Récupération des URLs des images
    image_urls = [result.url for result in gis.results()]

    # Boucle pour vérifier les URLs des images et récupérer la première valide
    for image_url in image_urls:
        try:
            # Timeout de 5 secondes pour la requête
            response = requests.get(image_url, timeout=5)
            if 'image' in response.headers.get('content-type', '').lower():
                return image_url  # Renvoie l'URL de la première image valide

        except requests.Timeout:
            print(f"Request timed out for URL: {image_url}")
            continue

        except Exception as e:
            print(f"Error fetching image: {e}")
            continue

    return None  # Renvoie None si aucune image valide n'est trouvée


# Fonction pour afficher une seule image à partir de son URL
def afficher_image(image_url):
    try:
        # Récupère l'image et redimensionne-la
        response = requests.get(image_url)
        if 'image' in response.headers.get('content-type', '').lower():
            image = Image.open(BytesIO(response.content))
            image = image.resize((500, 400))

            image_tk = ImageTk.PhotoImage(image)

            # Mettre à jour l'image dans le Label
            label_image_google.config(image=image_tk)
            label_image_google.image = image_tk  # Garde une référence pour éviter la suppression

    except Exception as e:
        print(f"Error displaying image: {e}")

# Fonction pour classer une image à l'aide du modèle MobileNetV2
def classify_image(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
        predictions)
    return decoded_predictions  # Récupère le libellé de la classe prédite


# Fonction pour charger une image à partir du système de fichiers
def load_image():
    try:
        # Sélectionne un fichier via une boîte de dialogue
        file_path = filedialog.askopenfilename()
        result = classify_image(file_path)

        # Supprime le texte des labels et les images affichées
        label_result.config(text="")
        label2.config(text="")
        label_image.config(image="")
        label_image_google.config(image="")

        # Clé d'API Google et CX
        api_key = 'AIzaSyBD-CG3JzojLSnuuK7hMyCq4oikp-EaSag'
        cx = '33123f43c8eb44299'

        # Spécifie le nom de l'image à rechercher
        query = result[0][0][1]

        # Récupère l'URL de l'image correspondante
        image_url = get_image_urls(api_key, cx, query)

        # Met à jour les labels avec les résultats
        label_result.config(
            text=f"Résultat 1: {result[0][0][1]} : {result[0][0][2]*100:.2f}% \nRésultat 2: {result[0][1][1]} : {result[0][1][2]*100:.2f}%")
        label2.config(text=f"Image correspondant: {query}")

        # Affiche l'image sous le bouton
        image = Image.open(file_path)
        image = image.resize((500, 400))
        image_tk = ImageTk.PhotoImage(image)
        label_image.config(image=image_tk)
        label_image.image = image_tk  # Garde une référence pour éviter la suppression

        if image_url:
            afficher_image(image_url)  # Affiche l'image s'il y en a une

    except Exception as e:
        print(f"Error loading image: {e}")


# Interface graphique
root = tk.Tk()
root.title("Classificateur d'images")
root.resizable(width=False, height=False)
root.geometry("700x850+0+0")  # Taille et position de la fenêtre

# Bouton pour charger une image
load_button = tk.Button(root, text="..load image", command=load_image)
load_button.pack()

# Label pour afficher les résultats de la classification
label_result = tk.Label(root, text="")
label_result.pack()

# Label pour afficher l'image chargée depuis le système de fichiers
label_image = tk.Label(root, image="")
label_image.pack()

# Label pour afficher le nom de l'image recherchée sur Google Images
label2 = tk.Label(root, text="")
label2.pack()

# Label pour afficher l'image provenant de Google Images Search
label_image_google = tk.Label(root, image="")
label_image_google.pack()

root.mainloop()
