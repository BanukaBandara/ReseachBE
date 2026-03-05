const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

const upload = multer({ dest: "uploads/" });

/**
 * POST /api/predict
 * FormData: image=<file>
 * Returns: { label: string, confidence: number }
 */
router.post("/", upload.single("image"), (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No image uploaded" });

    const imagePath = path.resolve(req.file.path);

    // ✅ python file + model paths
    const pyFile = path.resolve("ai-model/predict_api.py");

    // Run: python ai-model/predict_api.py <imagePath>
    const py = spawn("python", [pyFile, imagePath]);

    let stdout = "";
    let stderr = "";

    py.stdout.on("data", (chunk) => (stdout += chunk.toString()));
    py.stderr.on("data", (chunk) => (stderr += chunk.toString()));

    py.on("close", (code) => {
      // remove uploaded image after prediction
      fs.unlink(imagePath, () => {});

      if (code !== 0) {
        return res.status(500).json({
          error: "Prediction failed",
          details: stderr || "Python exited with error",
        });
      }

      try {
        const result = JSON.parse(stdout);
        return res.json(result);
      } catch (e) {
        return res.status(500).json({
          error: "Invalid model output (not JSON)",
          raw: stdout,
          details: stderr,
        });
      }
    });
  } catch (err) {
    return res.status(500).json({ error: err.message || "Server error" });
  }
});

module.exports = router;