const jwt = require('jsonwebtoken');
const User = require('../models/user_model');

const protect = (req, res, next) => {
  let token;

  if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
    token = req.headers.authorization.split(' ')[1];
  }

  if (!token) {
    return res.status(401).json({
      success: false,
      message: 'No token, authorization denied'
    });
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => {
    if (err) {
      return res.status(401).json({
        success: false,
        message: 'Token is not valid'
      });
    }

    User.findById(decoded.id)
      .then(user => {
        if (!user) {
          return res.status(401).json({
            success: false,
            message: 'User not found'
          });
        }

        req.user = user;
        next();
      })
      .catch(error => {
        res.status(500).json({
          success: false,
          message: 'Server error',
          error: error.message
        });
      });
  });
};

module.exports = { protect };