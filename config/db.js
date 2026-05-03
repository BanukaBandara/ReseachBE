const mongoose = require("mongoose");

const connectDB = async (options = {}) => {
  try {
    const uri = process.env.MONGO_URI;

    if (!uri) {
      console.log("⚠️ MONGO_URI not set in .env - running WITHOUT database");
      return;
    }

    // Connection settings update kara
    const connectionOptions = {
      ...options,
      family: 4, 
      serverSelectionTimeoutMS: 5000, 
    };

    const conn = await mongoose.connect(uri, connectionOptions);
    
    console.log(`✅ MongoDB Connected: ${conn.connection.host}`);
  } catch (err) {
    console.log("❌ MongoDB connection error:", err.message);

    // Error eka wistara karala pennanna
    if (err.message.includes("ECONNREFUSED")) {
      console.log("👉 Suggestion: Check if your IP is whitelisted (0.0.0.0/0) or change DNS to 8.8.8.8");
    }

    console.log("⚠️ Server will continue WITHOUT database (fix Mongo later)");
  }
};

module.exports = connectDB;