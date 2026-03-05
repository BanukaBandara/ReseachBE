import sys, os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pineapple_pest_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (224, 224)

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.exists(MODEL_PATH):
        print(json.dumps({"error": f"Model not found: {MODEL_PATH}"}))
        sys.exit(1)

    if not os.path.exists(CLASS_NAMES_PATH):
        print(json.dumps({"error": f"class_names.json not found: {CLASS_NAMES_PATH}"}))
        sys.exit(1)

    if not os.path.exists(img_path):
        print(json.dumps({"error": f"Image not found: {img_path}"}))
        sys.exit(1)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))

    if isinstance(class_names, list):
        label = class_names[idx]
    else:
        label = class_names[str(idx)]

    confidence = float(preds[idx])

    print(json.dumps({
        "label": label,
        "confidence": round(confidence * 100, 2)  # returns 0-100
    }))

if __name__ == "__main__":
    main()