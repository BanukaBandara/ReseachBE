const axios = require("axios");
const FormData = require("form-data");

const PYTHON_API_URL = "http://127.0.0.1:8000/predict"; // change to your python server IP if needed

exports.predictPest = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: "No image uploaded" });
    }

    const formData = new FormData();
    formData.append("file", req.file.buffer, {
      filename: req.file.originalname || "image.jpg",
      contentType: req.file.mimetype || "image/jpeg",
    });

    const response = await axios.post(PYTHON_API_URL, formData, {
      headers: formData.getHeaders(),
      timeout: 20000,
    });

    return res.json({
      success: true,
      ...response.data, // {label, confidence}
    });
  } catch (error) {
    console.error("Predict Error:", error.message);
    return res.status(500).json({
      success: false,
      message: "Prediction failed",
    });
  }
};
