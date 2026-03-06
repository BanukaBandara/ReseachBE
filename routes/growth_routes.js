const express = require("express");
const multer = require("multer");

const {
  predictGrowth,
  getGrowthHistory,
  getRegionalAlerts,
  submitGrowthFeedback,
  getVoiceAlert,
} = require("../controllers/growth_controller");

const router = express.Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    // Mobile photos can be large; keep a sane limit to prevent memory pressure
    fileSize: 8 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    const isImage = typeof file.mimetype === "string" && file.mimetype.startsWith("image/");
    cb(isImage ? null : new Error("Only image uploads are allowed"), isImage);
  },
});

router.post("/predict", upload.single("file"), predictGrowth);
router.get("/history/:farmerId", getGrowthHistory);
router.get("/alerts", getRegionalAlerts);
router.post("/feedback", submitGrowthFeedback);
router.post("/voice-alert/:reportId", getVoiceAlert);

module.exports = router;
