const axios = require("axios");
const FormData = require("form-data");
const http = require("http");
const https = require("https");
const mongoose = require("mongoose");

const GrowthReport = require("../models/growth_report_model");

const DEFAULT_PY_API = "http://127.0.0.1:8001/predict";
const PYTHON_API_URL = process.env.GROWTH_PYTHON_API_URL || DEFAULT_PY_API;

const axiosClient = axios.create({
  timeout: 60000,
  httpAgent: new http.Agent({ keepAlive: true }),
  httpsAgent: new https.Agent({ keepAlive: true }),
  maxContentLength: Infinity,
  maxBodyLength: Infinity,
});

const isDbReady = () => mongoose?.connection?.readyState === 1;

function buildRecommendation(condition, confidence) {
  const recommendations = {
    healthy: {
      status: "✅ HEALTHY PLANT",
      action: "Continue current care routine",
      details: [
        "Maintain regular watering schedule",
        "Keep fertilization consistent",
        "Monitor for early signs of stress",
        "Take photos weekly to track growth",
      ],
    },
    nitrogen_deficiency: {
      status: "⚠️ NITROGEN DEFICIENCY DETECTED",
      action: "Apply nitrogen-rich fertilizer",
      details: [
        "Apply urea or ammonium sulfate",
        "Water well after fertilizer application",
        "Re-check in 7–10 days for improvement",
        "Older leaves may not recover; watch new growth",
      ],
    },
    water_stress: {
      status: "💧 WATER STRESS DETECTED",
      action: "Adjust watering immediately",
      details: [
        "Increase watering frequency",
        "Check soil moisture before watering",
        "Water deeply but avoid waterlogging",
        "Add mulch to retain moisture",
      ],
    },
  };

  const rec = recommendations[condition] || {
    status: `ℹ️ Prediction: ${condition}`,
    action: "No predefined recommendation",
    details: ["Review the photo and try again with better lighting."],
  };

  let confidenceMessage = "";
  if (typeof confidence === "number") {
    if (confidence < 60) confidenceMessage = "⚠️ Low confidence — please retake photo";
    else if (confidence < 80) confidenceMessage = "🔸 Moderate confidence — monitor closely";
    else confidenceMessage = "✅ High confidence — follow the recommendation";
  }

  return { ...rec, confidenceMessage };
}

function buildVoiceText({ condition, month, language, recommendation }) {
  const lang = (language || "en").toLowerCase();

  const templates = {
    en: {
      healthy: `Your plant looks healthy. Month: ${month}. ${recommendation.action}.`,
      nitrogen_deficiency: `Nitrogen deficiency detected. Month: ${month}. ${recommendation.action}.`,
      water_stress: `Water stress detected. Month: ${month}. ${recommendation.action}.`,
      unknown: `Prediction: ${condition}. Month: ${month}. ${recommendation.action}.`,
    },
    si: {
      healthy: `ඔබගේ පැළ සෞඛ්‍ය සම්පන්නයි. මාසය: ${month}. ${recommendation.action}.`,
      nitrogen_deficiency: `නයිට්‍රජන් හිඟයක් හඳුනාගෙන ඇත. මාසය: ${month}. ${recommendation.action}.`,
      water_stress: `ජල අඩුවක් හඳුනාගෙන ඇත. මාසය: ${month}. ${recommendation.action}.`,
      unknown: `ප්‍රතිඵලය: ${condition}. මාසය: ${month}. ${recommendation.action}.`,
    },
    ta: {
      healthy: `உங்கள் செடி ஆரோக்கியமாக உள்ளது. மாதம்: ${month}. ${recommendation.action}.`,
      nitrogen_deficiency: `நைட்ரஜன் குறைபாடு கண்டறியப்பட்டது. மாதம்: ${month}. ${recommendation.action}.`,
      water_stress: `நீர் அழுத்தம் கண்டறியப்பட்டது. மாதம்: ${month}. ${recommendation.action}.`,
      unknown: `முடிவு: ${condition}. மாதம்: ${month}. ${recommendation.action}.`,
    },
  };

  const dict = templates[lang] || templates.en;
  return dict[condition] || dict.unknown;
}

function parseNumber(value) {
  if (value === undefined || value === null || value === "") return undefined;
  const num = Number(value);
  return Number.isFinite(num) ? num : undefined;
}

exports.predictGrowth = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: "No image uploaded" });
    }

    const farmerId = req.body.farmerId || req.body.farmer_id;
    const region = req.body.region;
    const language = req.body.language || "en";
    const lat = parseNumber(req.body.lat);
    const lon = parseNumber(req.body.lon);

    const formData = new FormData();
    formData.append("file", req.file.buffer, {
      filename: req.file.originalname || "image.jpg",
      contentType: req.file.mimetype || "image/jpeg",
    });

    const response = await axiosClient.post(PYTHON_API_URL, formData, {
      headers: formData.getHeaders(),
    });

    const data = response.data;
    if (!data || data.success === false) {
      return res.status(502).json({ success: false, message: "ML service failed", details: data });
    }

    const condition = data.condition;
    const month = data.month;
    const recommendation = buildRecommendation(condition, data.condition_confidence);
    const voiceText = buildVoiceText({
      condition,
      month,
      language,
      recommendation,
    });

    // If Mongo isn't configured/connected, don't block the request.
    // The app can still show the prediction without persisting history.
    let reportId = null;
    if (isDbReady()) {
      try {
        const report = await GrowthReport.create({
          farmerId,
          region,
          location: { lat, lon },
          language,
          voiceText,
          image: {
            originalName: req.file.originalname,
            mimeType: req.file.mimetype,
            sizeBytes: req.file.size,
          },
          prediction: {
            month,
            condition,
            predictedClass: data.predicted_class,
            confidence: data.confidence,
            probabilities: data.probabilities,
            modelVersion: data.model_version,
          },
          recommendation,
        });
        reportId = report?._id || null;
      } catch (e) {
        // Don't fail prediction if persistence fails
        reportId = null;
      }
    }

    const explain = String(req.query.explain || "").toLowerCase() === "true";

    const payload = {
      success: true,
      reportId,
      prediction: {
        month,
        condition,
        predictedClass: data.predicted_class,
        confidence: data.confidence,
        probabilities: data.probabilities,
        modelVersion: data.model_version,
      },
      recommendation,
      voiceText,
    };

    if (explain) {
      payload.explainability = {
        gradcam_overlay_png_base64: data.gradcam_overlay_png_base64,
      };
    }

    return res.json(payload);
  } catch (error) {
    console.error("Growth Predict Error:", error.message);
    return res.status(500).json({ success: false, message: "Growth prediction failed" });
  }
};

exports.getGrowthHistory = async (req, res) => {
  try {
    if (!isDbReady()) {
      return res
        .status(503)
        .json({ success: false, message: "Database unavailable (set MONGO_URI)" });
    }
    const { farmerId } = req.params;
    const limit = Math.min(Number(req.query.limit || 20), 100);

    const rows = await GrowthReport.find({ farmerId })
      .sort({ createdAt: -1 })
      .limit(limit)
      .select({
        prediction: 1,
        recommendation: 1,
        region: 1,
        language: 1,
        voiceText: 1,
        createdAt: 1,
        feedback: 1,
      });

    return res.json({ success: true, items: rows });
  } catch (error) {
    console.error("Growth History Error:", error.message);
    return res.status(500).json({ success: false, message: "Failed to load history" });
  }
};

exports.getRegionalAlerts = async (req, res) => {
  try {
    if (!isDbReady()) {
      return res
        .status(503)
        .json({ success: false, message: "Database unavailable (set MONGO_URI)" });
    }
    const region = req.query.region;
    const days = Math.min(Number(req.query.days || 7), 60);
    const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

    const match = { createdAt: { $gte: since } };
    if (region) match.region = region;

    const agg = await GrowthReport.aggregate([
      { $match: match },
      {
        $group: {
          _id: { region: "$region", condition: "$prediction.condition" },
          count: { $sum: 1 },
        },
      },
      { $sort: { count: -1 } },
    ]);

    const summary = {};
    for (const row of agg) {
      const r = row._id.region || "unknown";
      const c = row._id.condition || "unknown";
      if (!summary[r]) summary[r] = {};
      summary[r][c] = row.count;
    }

    // Very simple heuristic: flag if >= 3 reports of same issue
    const alerts = [];
    for (const [r, counts] of Object.entries(summary)) {
      const nitrogen = counts.nitrogen_deficiency || 0;
      const water = counts.water_stress || 0;
      if (nitrogen >= 3) alerts.push({ region: r, issue: "nitrogen_deficiency", count: nitrogen });
      if (water >= 3) alerts.push({ region: r, issue: "water_stress", count: water });
    }

    return res.json({ success: true, since, summary, alerts });
  } catch (error) {
    console.error("Regional Alerts Error:", error.message);
    return res.status(500).json({ success: false, message: "Failed to load alerts" });
  }
};

exports.submitGrowthFeedback = async (req, res) => {
  try {
    if (!isDbReady()) {
      return res
        .status(503)
        .json({ success: false, message: "Database unavailable (set MONGO_URI)" });
    }
    const { reportId, isCorrect, correctCondition, notes } = req.body || {};

    if (!reportId) {
      return res.status(400).json({ success: false, message: "reportId is required" });
    }

    const report = await GrowthReport.findById(reportId);
    if (!report) {
      return res.status(404).json({ success: false, message: "Report not found" });
    }

    report.feedback = {
      isCorrect: typeof isCorrect === "boolean" ? isCorrect : undefined,
      correctCondition,
      notes,
    };

    await report.save();

    return res.json({ success: true });
  } catch (error) {
    console.error("Feedback Error:", error.message);
    return res.status(500).json({ success: false, message: "Failed to submit feedback" });
  }
};

exports.getVoiceAlert = async (req, res) => {
  try {
    if (!isDbReady()) {
      return res
        .status(503)
        .json({ success: false, message: "Database unavailable (set MONGO_URI)" });
    }
    const { reportId } = req.params;
    const { language } = req.body || {};

    if (!reportId) {
      return res.status(400).json({ success: false, message: "reportId is required" });
    }

    const report = await GrowthReport.findById(reportId).select({
      voiceText: 1,
      language: 1,
      prediction: 1,
      recommendation: 1,
    });

    if (!report) {
      return res.status(404).json({ success: false, message: "Report not found" });
    }

    const voiceText =
      report.voiceText ||
      buildVoiceText({
        condition: report?.prediction?.condition,
        month: report?.prediction?.month,
        language: language || report.language || "en",
        recommendation: report.recommendation || { action: "" },
      });

    return res.json({ success: true, message_text: voiceText });
  } catch (error) {
    console.error("Voice Alert Error:", error.message);
    return res.status(500).json({ success: false, message: "Failed to generate voice alert" });
  }
};
