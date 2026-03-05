const axios = require("axios");
const FormData = require("form-data");

const PYTHON_API = process.env.PYTHON_API || "http://192.168.8.181:8000";

exports.predictDisease = async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ message: "image is required" });

    const form = new FormData();
    form.append("image", req.file.buffer, {
      filename: req.file.originalname || "image.jpg",
      contentType: req.file.mimetype || "image/jpeg",
    });

    const resp = await axios.post(`${PYTHON_API}/predict/disease`, form, {
      headers: form.getHeaders(),
      timeout: 60000,
    });

    const data = resp.data;
    const conf = data.confidence || 0;

    // optional severity logic
    let severity = "Low";
    if (conf >= 0.85) severity = "High";
    else if (conf >= 0.65) severity = "Medium";

    return res.json({ ...data, severity });
  } catch (e) {
    console.log("predictDisease error:", e?.response?.data || e.message);
    return res.status(500).json({
      message: "Disease prediction failed",
      error: e?.response?.data || e.message,
    });
  }
};