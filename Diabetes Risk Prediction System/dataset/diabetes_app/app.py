from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import numpy as np

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")
model = pickle.load(open(model_path, "rb"))

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

# ---------------- BLOG ----------------
@app.route("/blog")
def blog():
    return render_template("blog.html")

@app.route("/blog/early-signs")
def early_signs():
    return render_template("early_signs.html")

@app.route("/blog/ai-diabetes")
def ai_diabetes():
    return render_template("ai_diabetes.html")

@app.route("/blog/diet")
def diet():
    return render_template("diet.html")

@app.route("/blog/prevention")
def prevention():
    return render_template("prevention.html")

@app.route("/blog/lifestyle")
def lifestyle():
    return render_template("lifestyle.html")

# ---------------- BMI ----------------
@app.route("/bmi", methods=["GET","POST"])
def bmi():
    bmi_value = None
    category = None
    weight = None
    height = None
    error = None

    if request.method == "POST":
        try:
            weight = float(request.form.get("weight", 0))
            height = float(request.form.get("height", 0))

            if height <= 0 or weight <= 0:
                error = "Enter valid numbers"
            else:
                bmi_value = round(weight / ((height/100)**2), 1)

                if bmi_value < 18.5:
                    category = "Underweight"
                elif bmi_value < 25:
                    category = "Normal"
                elif bmi_value < 30:
                    category = "Overweight"
                else:
                    category = "Obese"
        except:
            error = "Invalid input"

    return render_template(
        "bmi.html",
        bmi_value=bmi_value,
        category=category,
        weight=weight,
        height=height,
        error=error
    )

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        try:
            f = request.form

            # Validate BMI is provided
            bmi_val = f.get("BMI", "").strip()
            if not bmi_val:
                return render_template("predict.html", error="Please enter your BMI value in Step 4.")

            form_data = f.to_dict()

            # Feature order must match training data
            data = [[
                float(f.get("HighBP", 0)),
                float(f.get("HighChol", 0)),
                float(f.get("CholCheck", 0)),
                float(bmi_val),
                float(f.get("Smoker", 0)),
                float(f.get("Stroke", 0)),
                float(f.get("HeartDiseaseorAttack", 0)),
                float(f.get("PhysActivity", 0)),
                float(f.get("Fruits", 0)),
                float(f.get("Veggies", 0)),
                float(f.get("HvyAlcoholConsump", 0)),
                float(f.get("AnyHealthcare", 0)),
                float(f.get("NoDocbcCost", 0)),
                float(f.get("GenHlth", 3)),
                float(f.get("MentHlth", 0)),
                float(f.get("PhysHlth", 0)),
                float(f.get("DiffWalk", 0)),
                float(f.get("Sex", 1)),
                float(f.get("Age", 1)),
                float(f.get("Education", 4)),
                float(f.get("Income", 5))
            ]]

            data = np.array(data)
            prediction = model.predict(data)[0]
            prob = model.predict_proba(data)[0][1]

            risk_score = round(prob * 100, 1)
            if risk_score < 30:
                risk_level = "Low"
            elif risk_score < 60:
                risk_level = "Moderate"
            else:
                risk_level = "High"

            feature_scores = {
                "Blood Pressure": float(f.get("HighBP", 0)) * 80 + 20,
                "Cholesterol": float(f.get("HighChol", 0)) * 70 + 15,
                "Physical Activity": (1 - float(f.get("PhysActivity", 0))) * 60 + 20,
                "Diet Quality": (1 - (float(f.get("Fruits", 0)) + float(f.get("Veggies", 0))) / 2) * 70 + 15,
                "BMI Risk": min(100, max(0, (float(bmi_val) - 18.5) / 25 * 100)),
                "Lifestyle": (float(f.get("Smoker", 0)) * 40 + float(f.get("HvyAlcoholConsump", 0)) * 40 + 20),
            }

            return render_template(
                "result.html",
                risk_score=risk_score,
                risk_level=risk_level,
                prediction=int(prediction),
                probability=round(prob * 100, 2),
                bmi=float(bmi_val),
                age_group=f.get("Age", "1"),
                sex=f.get("Sex", "1"),
                feature_scores=feature_scores,
                form_data=form_data
            )

        except Exception as e:
            print("PREDICTION ERROR:", e)
            import traceback
            traceback.print_exc()
            return render_template("predict.html", error=f"Something went wrong: {str(e)}")

    return render_template("predict.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)