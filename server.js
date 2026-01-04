require("dotenv").config();
const express = require("express");
const cors = require("cors");
const connectDB = require("./config/db");

const app = express();

// ✅ MUST HAVE for JSON body
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ✅ CORS
app.use(cors({ origin: "*", credentials: true }));

// ✅ Connect DB
connectDB();

// ✅ Routes (must match your frontend)
app.use("/api/auth", require("./routes/user_routes"));

// Health check
app.get("/", (req, res) => res.send("✅ Backend running"));
app.get("/api/health", (req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
