from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

modelo = joblib.load("modelo_lr.pkl")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
def predict(request: Request,
            Age: float = Form(...),
            RestingBP: float = Form(...),
            Cholesterol: float = Form(...),
            MaxHR: float = Form(...),
            Oldpeak: float = Form(...),
            Sex: str = Form(...),
            FastingBS: int = Form(...),
            ChestPainType: str = Form(...),
            RestingECG: str = Form(...),
            ExerciseAngina: str = Form(...),
            ST_Slope: str = Form(...)):

    input_dict = {
        'Age': [Age],
        'RestingBP': [RestingBP],
        'Cholesterol': [Cholesterol],
        'MaxHR': [MaxHR],
        'Oldpeak': [Oldpeak],
        'Sex': [Sex],
        'FastingBS': [FastingBS],
        'ChestPainType': [ChestPainType],
        'RestingECG': [RestingECG],
        'ExerciseAngina': [ExerciseAngina],
        'ST_Slope': [ST_Slope]
    }

    df = pd.DataFrame(input_dict)
    pred = modelo.predict(df)[0]
    prob = modelo.predict_proba(df)[0][1]

    resultado = "Riesgo cardíaco detectado" if pred == 1 else "Sin riesgo cardíaco"
    return templates.TemplateResponse("form.html", {
        "request": request,
        "resultado": resultado,
        "probabilidad": f"{prob:.2%}"
    })
