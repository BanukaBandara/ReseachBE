const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

const upload = multer({ dest: "uploads/" });

/**
 * POST /api/disease/predict
 * FormData: image=<file>
 */
router.post("/predict", upload.single("image"), (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No image uploaded" });

    const imagePath = path.resolve(req.file.path);

    const pyFile = path.resolve("ai-model/predict_disease_api.py");

    const py = spawn("python", [pyFile, imagePath]);

    let stdout = "";
    let stderr = "";

    py.stdout.on("data", (chunk) => (stdout += chunk.toString()));
    py.stderr.on("data", (chunk) => (stderr += chunk.toString()));

    py.on("close", (code) => {
      fs.unlink(imagePath, () => {});

      if (code !== 0) {
        return res.status(500).json({
          error: "Disease prediction failed",
          details: stderr || "Python exited with error",
        });
      }

      try {
        const result = JSON.parse(stdout);
        return res.json(result);
      } catch (e) {
        return res.status(500).json({
          error: "Invalid JSON from Python",
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