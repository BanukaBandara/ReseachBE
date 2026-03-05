import sys, os, json
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "pineapple_disease_detector_final.h5")
CLASS_JSON = os.path.join(BASE_DIR, "disease_class_names.json")
IMG_SIZE = (224, 224)

def load_class_names(p):
    with open(p, "r") as f:
        data = json.load(f)

    # case 1: list -> already ordered
    if isinstance(data, list):
        return data

    # case 2: dict:
    # 2a) {"0":"Crown Rot", "1":"Fruit Rot"}  (index->label)
    if all(str(k).isdigit() for k in data.keys()):
        return [data[str(i)] for i in range(len(data))]

    # 2b) {"Crown Rot":0, "Fruit Rot":1} (label->index)
    # sort by index values
    return [k for k, v in sorted(data.items(), key=lambda x: x[1])]

def severity_from_conf(c01: float):
    if c01 >= 0.80:
        return "High"
    if c01 >= 0.60:
        return "Medium"
    return "Low"

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)

    img_path = sys.argv[1]

    if not os.path.exists(MODEL_PATH):
        print(json.dumps({"error": f"Model not found: {MODEL_PATH}"}))
        sys.exit(1)

    if not os.path.exists(CLASS_JSON):
        print(json.dumps({"error": f"class_names.json not found: {CLASS_JSON}"}))
        sys.exit(1)

    if not os.path.exists(img_path):
        print(json.dumps({"error": f"Image not found: {img_path}"}))
        sys.exit(1)

    class_names = load_class_names(CLASS_JSON)

    model = tf.keras.models.load_model(MODEL_PATH)

    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = class_names[idx] if idx < len(class_names) else "Unknown"
    conf01 = float(preds[idx])

    print(json.dumps({
        "disease": label,
        "confidence": round(conf01 * 100, 2),  # 0-100
        "severity": severity_from_conf(conf01),
        "notes": []
    }))

if __name__ == "__main__":
    main()