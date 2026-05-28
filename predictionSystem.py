import pandas as pd
import joblib

model = joblib.load("random_forest_model.pkl")


def predict_pest_risk(form_data):

    try:
        input_data = pd.DataFrame([{
            "department": form_data["department"],
            "municipality": int(form_data["municipality"]),
            "crop": form_data["crop"],
            "year": form_data["year"],
            "season": form_data["season"],
            "pest_name": form_data["pest_name"],
            "species": form_data["species"],
            "yield_t_ha": float(form_data["yield_t_ha"]),
            "relative_yield": float(form_data["relative_yield"]),
            "avg_temperature": float(form_data["avg_temperature"]),
            "total_precipitation": float(form_data["total_precipitation"]),
            "climate_risk": int(
                float(form_data["avg_temperature"]) > 28 and
                float(form_data["total_precipitation"]) > 100
            ),
            "category_temp": 1 if float(form_data["avg_temperature"]) > 20 else 0,
            "category_rain": 0
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        confidence = round(max(probabilities) * 100, 2)

        propagation_probability = probabilities[1]

        if propagation_probability < 0.40:
            risk_level = "Low Propagation Risk"
            description = "Environmental conditions show low probability of pest propagation."

        elif propagation_probability < 0.70:
            risk_level = "Medium Propagation Risk"
            description = "Environmental conditions suggest moderate pest risk."

        else:
            risk_level = "High Propagation Risk"
            description = "Environmental conditions indicate high pest propagation risk."

        return {
            "prediction": risk_level,
            "confidence": confidence,
            "description": description,
            "probabilities": [
                {"label": "No Alert", "value": round(probabilities[0] * 100, 2)},
                {"label": "Alert", "value": round(probabilities[1] * 100, 2)}
            ]
        }

    except Exception as e:
        return {"error": str(e)}