from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os

app = FastAPI(title="ML Model API", description="API для доступа к модели ML")

# Путь к папке с моделями
MODEL_DIR = "models"
MODEL_FILENAME = "random_forest_model.pkl"

# Путь к файлу с названиями признаков
FEATURE_NAMES_PATH = "feature_names.pkl"

# Загрузка модели
try:
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден по пути {model_path}")
    model = joblib.load(model_path)
    print("Модель успешно загружена")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Ошибка при загрузке модели: {str(e)}")

# Загрузка названий признаков
try:
    with open(os.path.join(MODEL_DIR, FEATURE_NAMES_PATH), 'rb') as f:
        feature_names = joblib.load(f)
    print(f"Названия признаков загружены: {feature_names[:5]}...")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Файл с названиями признаков не найден")

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI Docker API!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: dict):
    """
    Предсказание модели на основе входных данных
    """
    try:
        # Создаем DataFrame из входных данных
        input_data = pd.DataFrame([data], columns=feature_names)

        # Проверяем, что все признаки присутствуют
        missing_features = set(feature_names) - set(input_data.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Не хватает следующих признаков: {missing_features}"
            )

        # Делаем предсказание
        prediction = model.predict(input_data)
        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)