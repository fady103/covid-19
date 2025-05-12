import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

__version__ = "1.0.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# تحميل النموذج بصيغة .h5
model_path = f"{BASE_DIR}/covid_model.h5"
model = load_model(model_path)

def predict_pipeline(image_path):
    try:
        # تحميل الصورة ومعالجتها
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # توقع الفئة (COVID or Normal)
        prediction = model.predict(img)[0][0]
        label = "COVID" if prediction > 0.5 else "Normal"
        return label
    except Exception as e:
        return f"Error: {str(e)}"
