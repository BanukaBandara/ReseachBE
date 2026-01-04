const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const User = require("../models/user_model");

const makeToken = (id) => {
  return jwt.sign({ id }, process.env.JWT_SECRET, { expiresIn: "7d" });
};

exports.register = async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // ✅ Validate
    if (!name || !email || !password) {
      return res.status(400).json({
        success: false,
        message: "name, email, password are required",
      });
    }

    const cleanEmail = email.toLowerCase().trim();

    const existing = await User.findOne({ email: cleanEmail });
    if (existing) {
      return res.status(409).json({
        success: false,
        message: "Email already exists",
      });
    }

    // ✅ Hash
    const hashed = await bcrypt.hash(password, 10);

    // ✅ Create user
    const user = await User.create({
      name: name.trim(),
      email: cleanEmail,
      password: hashed,
    });

    // ✅ Create token
    const token = makeToken(user._id);

    return res.status(201).json({
      success: true,
      message: "User registered successfully",
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
      },
    });
  } catch (err) {
    console.error("REGISTER ERROR:", err);
    return res.status(500).json({
      success: false,
      message: "Server error in register",
    });
  }
};

exports.login = async (req, res) => {
  try {
    const { email, password } = req.body;

    // ✅ Validate
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: "email and password are required",
      });
    }

    const cleanEmail = email.toLowerCase().trim();

    // ✅ Find user
    const user = await User.findOne({ email: cleanEmail });
    if (!user) {
      return res.status(401).json({
        success: false,
        message: "Invalid email or password",
      });
    }

    // ✅ Compare password
    const ok = await bcrypt.compare(password, user.password);
    if (!ok) {
      return res.status(401).json({
        success: false,
        message: "Invalid email or password",
      });
    }

    // ✅ Token
    const token = makeToken(user._id);

    return res.json({
      success: true,
      message: "Login successful",
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
      },
    });
  } catch (err) {
    console.error("LOGIN ERROR:", err);
    return res.status(500).json({
      success: false,
      message: "Server error in login",
    });
  }
};
