from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from project.utils import load_object
from project.utils.feature_engineering import preprocess_customer_data
from project.entity.estimator import ProjectModel

app = FastAPI()
templates = Jinja2Templates(directory="project/templates")

# Load your trained model (already wrapped as ProjectModel if needed)
model = load_object("final_model/prediction_model/pred_model.pkl")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    CreditScore: float = Form(...),
    Age: float = Form(...),
    Tenure: int = Form(...),
    Balance: float = Form(...),
    NumOfProducts: int = Form(...),
    HasCrCard: int = Form(...),
    IsActiveMember: int = Form(...),
    EstimatedSalary: float = Form(...),
    Geography: str = Form(...),
    Gender: str = Form(...)
):
    try:
        # Convert form inputs into a dictionary
        input_data = {
            "CreditScore": float(CreditScore),
            "Age": float(Age),
            "Tenure": int(Tenure),
            "Balance": float(Balance),
            "NumOfProducts": int(NumOfProducts),
            "HasCrCard": int(HasCrCard),
            "IsActiveMember": int(IsActiveMember),
            "EstimatedSalary": float(EstimatedSalary),
            "Geography": Geography.strip(),
            "Gender": Gender.strip()
        }

        # Preprocess the data
        processed_data = preprocess_customer_data(input_data)

        # Make prediction using the trained model
        prediction_df = model.predict(processed_data)
        prediction = prediction_df['prediction'].iloc[0]

        # Convert numeric prediction into a readable message
        result = "Customer Will Exit" if prediction == 1 else "Customer Will Stay"

    except Exception as e:
        result = f"Error: {e}"

    # Render the same template with the prediction result
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


## uvicorn app:app --reload
