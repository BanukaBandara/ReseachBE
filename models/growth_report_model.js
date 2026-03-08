const mongoose = require("mongoose");

const locationSchema = new mongoose.Schema(
  {
    lat: { type: Number },
    lon: { type: Number },
  },
  { _id: false }
);

const predictionSchema = new mongoose.Schema(
  {
    month: { type: String },
    condition: { type: String },
    predictedClass: { type: String },
    confidence: { type: Number },
    probabilities: { type: Object },
    modelVersion: { type: String },
  },
  { _id: false }
);

const recommendationSchema = new mongoose.Schema(
  {
    status: { type: String },
    action: { type: String },
    details: { type: [String], default: [] },
    confidenceMessage: { type: String },
  },
  { _id: false }
);

const feedbackSchema = new mongoose.Schema(
  {
    isCorrect: { type: Boolean },
    correctCondition: { type: String },
    notes: { type: String },
  },
  { _id: false }
);

const growthReportSchema = new mongoose.Schema(
  {
    farmerId: { type: String, trim: true },
    region: { type: String, trim: true },
    location: { type: locationSchema },

    image: {
      originalName: { type: String },
      mimeType: { type: String },
      sizeBytes: { type: Number },
    },

    prediction: { type: predictionSchema },
    recommendation: { type: recommendationSchema },

    language: { type: String, default: "en" },
    voiceText: { type: String },

    feedback: { type: feedbackSchema },
  },
  { timestamps: true }
);

module.exports = mongoose.model("GrowthReport", growthReportSchema);
