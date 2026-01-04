const express = require("express");
const router = express.Router();

const { register, login } = require("../controllers/user_controller");

// ✅ These paths must match your frontend exactly:
router.post("/user/register", register);
router.post("/user/login", login);

module.exports = router;
