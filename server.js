require("dotenv").config();
const express = require("express");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const connectDB = require("./config/db");

const app = express();

// ---------- Middlewares ----------
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

app.use(
  cors({
    origin: "*",
    credentials: true,
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// ---------- Ensure uploads folder exists ----------
const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Optional: serve uploads (if you ever want to access files by URL)
app.use("/uploads", express.static(uploadsDir));

// ---------- Connect DB (non-blocking) ----------
connectDB().catch((err) => {
  console.log("⚠️ DB connect promise rejected:", err?.message || err);
});

// ---------- Routes ----------
app.use("/api/auth", require("./routes/user_routes"));
app.use("/api/predict", require("./routes/predict_routes")); // pest
app.use("/api/disease", require("./routes/disease_predict_routes")); // disease

// ---------- Health & Debug ----------
app.get("/", (req, res) => res.send("✅ Backend running"));
app.get("/api/health", (req, res) => res.json({ ok: true }));

// Helpful for testing
app.get("/api/routes", (req, res) => {
  res.json({
    health: "/api/health",
    pestPredict: "POST /api/predict  (FormData key: image)",
    diseasePredict: "POST /api/disease/predict  (FormData key: image)",
  });
});

// ---------- Start server ----------
const PORT = Number(process.env.PORT) || 3001;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${PORT}`);
});