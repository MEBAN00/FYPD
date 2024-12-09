from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.compose import ColumnTransformer
from pydantic import BaseModel
import pandas as pd
import pickle
import logging
from motor.motor_asyncio import AsyncIOMotorClient

# Initialize logging
logging.basicConfig(level=logging.INFO)



# Load the model
try:
    dt_model = pickle.load(open("ACTIV/project_model.pkl", "rb"))
    column_transformer = pickle.load(open("ACTIV/column_transformer.pkl", "rb"))
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    dt_model = None


# Create the FastAPI app
app = FastAPI()


# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define the data structure expected from the frontend
class PredictionInput(BaseModel):
    product: str
    quantity: int
    unit_price: float
    total_cost: float
    total_price: float
    Year: int


@app.post("/predict")
async def predict(new_data: PredictionInput):
    if dt_model is None:
        logging.error("Model is not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert the input data to a DataFrame
        new_df = pd.DataFrame([new_data.dict()])
        new_data_transformed = column_transformer.transform(new_df)
        pred = dt_model.predict(new_data_transformed)
        

        # Prepare response
        return JSONResponse(content={
            "prediction": int(pred[0])
        })
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction")

# Custom exception handler for validation errors
@app.exception_handler(HTTPException)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )