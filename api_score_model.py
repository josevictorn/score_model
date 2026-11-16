from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

app = FastAPI()

class request_body(BaseModel):
  study_hours: float

score_model = joblib.load('./score_model.pkl')


@app.post('/predict')
def predict(data: request_body):
  input_feature = [[data.study_hours]]
  y_pred = score_model.predict(input_feature)[0].astype(int)

  return {'score': y_pred.tolist()}