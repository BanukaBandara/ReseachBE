const mongoose = require("mongoose");

const connectDB = async () => {
  try {
    const uri = process.env.MONGO_URI;
    if (!uri) {
      console.log("⚠️ MONGO_URI not set - running WITHOUT database");
      return;
    }

    await mongoose.connect(uri);
    console.log("✅ MongoDB Connected:", mongoose.connection.host);
  } catch (err) {
    console.log("❌ MongoDB connection error:", err.message);
    console.log("⚠️ Server will continue WITHOUT database (fix Mongo later)");
  }
};

module.exports = connectDB;