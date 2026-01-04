const express = require("express");
const router = express.Router();
const multer = require("multer");
const { predictPest } = require("../controllers/pest_controller");

const upload = multer({ storage: multer.memoryStorage() });

router.post("/predict", upload.single("file"), predictPest);

module.exports = router;
