import requests
import pandas as pd
import numpy as np
import joblib
import gradio as gr


# Clés API
GEOCODIO_API_KEY = "0f1691619173ed9a137331313b36ad3373d1313"
CENSUS_API_KEY = "6389ac9239675526789757ece587332ea94c79a"

# Fonctions API
def get_lat_lng(city_name, neighborhood):
    query = f"{neighborhood}, {city_name}"
    url = "https://api.geocod.io/v1.7/geocode"
    params = {"q": query, "api_key": GEOCODIO_API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    results = response.json()["results"][0]["location"]
    return results["lat"], results["lng"]

def get_census_data(state_fips="06"):
    population = 1170
    median_income = 4
    return population, median_income

# Fonction principale pour Gradio
def generer_csv_et_prediction(rooms, bedrooms, house_age, city, neighborhood, occupation_input):
    try:
        # Enregistrement CSV
        df_csv = pd.DataFrame([[rooms, bedrooms, house_age, city, neighborhood, occupation_input]],
                              columns=["Rooms", "Bedrooms", "House_Age", "City", "Neighborhood", "occupation"])
        df_csv.to_csv("model_input_data.csv", index=False)

        # Récupération coordonnées
        lat, lng = get_lat_lng(city, neighborhood)

        # Population et income
        population, median_income = get_census_data("06")

        # Création DataFrame avec features brutes
        df = pd.DataFrame([[house_age, rooms, bedrooms, occupation_input, lat, lng, population, median_income]],
                          columns=["HouseAge","AveRooms","AveBedrms","AveOccup","Latitude","Longitude","Population","MedInc"])

        # Features dérivées
        df["MedHouseVal_log"] = 0  # placeholder pour GeoAxis
        df["MedInc_log1p"] = np.log1p(df["MedInc"])
        df["AveRooms_log1p"] = np.log1p(df["AveRooms"])
        df["AveBedrms_log1p"] = np.log1p(df["AveBedrms"])
        df["Population_log1p"] = np.log1p(df["Population"])
        df["AveOccup_log1p"] = np.log1p(df["AveOccup"])
        df["Rooms_per_occupant"] = df["AveRooms"] / df["AveOccup"]
        df["Bedrooms_per_room"] = df["AveBedrms"] / df["AveRooms"]

        # GeoAxis
        geo_model = joblib.load("model/GeoAxisModel.joblib")
        df["GeoAxis"] = geo_model.predict(df[["Latitude","Longitude"]].values)

        # Interactions
        df["MedInc_times_GeoAxis"] = df["MedInc"] * df["GeoAxis"]
        

        # Préparation X_new
        feature_order = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
                         'Latitude', 'Longitude', 'GeoAxis', 'MedInc_log1p', 'AveRooms_log1p', 'AveBedrms_log1p',
                         'Population_log1p', 'AveOccup_log1p', 'Rooms_per_occupant', 'Bedrooms_per_room',
                         'MedInc_times_GeoAxis']
        X_new = df[feature_order].values

        # Chargement modèle
        model_loaded = joblib.load("model/RandomForest.joblib")
        scaler_loaded = joblib.load("model/Scaler.joblib")
        X_new_scaled = scaler_loaded.transform(X_new)

        # Prédiction
        y_pred_log = model_loaded.predict(X_new_scaled)
        y_pred_real = np.expm1(y_pred_log) * 100000

        return f"Prédiction du prix moyen de la maison pour {neighborhood}, {city} : ${y_pred_real[0]:,.2f}"

    except Exception as e:
        return f"⚠ Une erreur est survenue : {e}"

# Interface Gradio
with gr.Blocks() as demo:
    rooms_input = gr.Number(label="Nombre de chambres (Rooms)", value=3)
    bedrooms_input = gr.Number(label="Nombre de chambres à coucher (Bedrooms)", value=2)
    house_age_input = gr.Number(label="Âge de la maison (House Age)", value=10)
    occupation_input = gr.Number(label="Occupation", value=10)
    city_input = gr.Textbox(label="Ville (ex: Los Angeles, CA)", value="Los Angeles, CA")
    neighborhood_input = gr.Textbox(label="Quartier (ex: Hollywood)", value="Hollywood")

    btn = gr.Button("Enregistrer et prédire", variant="primary")
    output_text = gr.Markdown()

    btn.click(
        generer_csv_et_prediction,
        inputs=[rooms_input, bedrooms_input, house_age_input, city_input, neighborhood_input, occupation_input],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()