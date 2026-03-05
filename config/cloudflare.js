const AWS = require('aws-sdk');
const crypto = require('crypto');
const path = require('path');
const https = require('https');

// Create custom HTTPS agent to handle SSL issues
const httpsAgent = new https.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 50,
  // Try different SSL/TLS settings
  secureProtocol: 'TLSv1_2_method', // Force TLS 1.2
  ciphers: 'ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS',
  honorCipherOrder: true,
  // Disable certificate validation if needed (NOT recommended for production)
  // rejectUnauthorized: false
});

// Configure AWS SDK for Cloudflare R2
const s3 = new AWS.S3({
  endpoint: `https://${process.env.CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com`,
  accessKeyId: process.env.CLOUDFLARE_R2_ACCESS_KEY,
  secretAccessKey: process.env.CLOUDFLARE_R2_SECRET_KEY,
  region: 'auto', // Changed back to 'auto' for Cloudflare R2
  signatureVersion: 'v4',
  s3ForcePathStyle: false, // Changed to false for R2
  sslEnabled: true,
  httpOptions: {
    timeout: 60000, // Increased to 60 seconds
    agent: httpsAgent, // Use custom HTTPS agent
    // Additional retry configuration
    maxRetries: 3,
    retryDelayOptions: {
      customBackoff: function(retryCount) {
        return Math.pow(2, retryCount) * 100; // Exponential backoff
      }
    }
  },
  // Additional S3 specific options
  computeChecksums: true,
  convertResponseTypes: false,
  correctClockSkew: true,
  maxRedirects: 10,
  paramValidation: true,
  s3DisableBodySigning: false,
  s3UsEast1RegionalEndpoint: 'regional'
});

const BUCKET_NAME = process.env.CLOUDFLARE_R2_BUCKET_NAME;
const PUBLIC_URL = process.env.CLOUDFLARE_R2_PUBLIC_URL;

// Generate unique filename
const generateFileName = (originalName, folder) => {
  const timestamp = Date.now();
  const random = crypto.randomBytes(6).toString('hex');
  const ext = path.extname(originalName).toLowerCase();
  return `${folder}/${timestamp}-${random}${ext}`;
};

// Validate image file
const validateImageFile = (file) => {
  const allowedMimes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  const maxSize = 5 * 1024 * 1024; // 5MB

  if (!allowedMimes.includes(file.mimetype)) {
    throw new Error('Invalid file type. Only JPEG, PNG, and WebP are allowed.');
  }

  if (file.size > maxSize) {
    throw new Error('File too large. Maximum size is 5MB.');
  }

  return true;
};

// Test connection to R2
const testConnection = async () => {
  try {
    console.log('Testing R2 connection...');
    const result = await s3.listObjects({
      Bucket: BUCKET_NAME,
      MaxKeys: 1
    }).promise();
    console.log('R2 connection test successful');
    return true;
  } catch (error) {
    console.error('R2 connection test failed:', error);
    return false;
  }
};

// Upload image to Cloudflare R2 with retry logic
exports.uploadImageToCloudflare = async (file, folder = 'images') => {
  try {
    // Environment validation
    if (!process.env.CLOUDFLARE_ACCOUNT_ID) {
      throw new Error('CLOUDFLARE_ACCOUNT_ID environment variable is not set');
    }
    if (!process.env.CLOUDFLARE_R2_ACCESS_KEY) {
      throw new Error('CLOUDFLARE_R2_ACCESS_KEY environment variable is not set');
    }
    if (!process.env.CLOUDFLARE_R2_SECRET_KEY) {
      throw new Error('CLOUDFLARE_R2_SECRET_KEY environment variable is not set');
    }
    if (!BUCKET_NAME) {
      throw new Error('CLOUDFLARE_R2_BUCKET_NAME environment variable is not set');
    }

    // Debug logging
    console.log('Environment check:', {
      hasAccountId: !!process.env.CLOUDFLARE_ACCOUNT_ID,
      hasAccessKey: !!process.env.CLOUDFLARE_R2_ACCESS_KEY,
      hasSecretKey: !!process.env.CLOUDFLARE_R2_SECRET_KEY,
      hasBucketName: !!BUCKET_NAME,
      hasPublicUrl: !!PUBLIC_URL,
      endpoint: `https://${process.env.CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com`
    });

    // Test connection first
    const connectionTest = await testConnection();
    if (!connectionTest) {
      throw new Error('Cannot establish connection to Cloudflare R2');
    }

    // Validate the file
    validateImageFile(file);

    // Generate unique filename
    const fileName = generateFileName(file.originalname, folder);

    // Upload parameters
    const uploadParams = {
      Bucket: BUCKET_NAME,
      Key: fileName,
      Body: file.buffer,
      ContentType: file.mimetype,
      CacheControl: 'max-age=31536000',
      ContentDisposition: 'inline',
      Metadata: {
        originalName: file.originalname,
        uploadedAt: new Date().toISOString(),
      }
    };

    console.log('Upload params:', {
      Bucket: uploadParams.Bucket,
      Key: uploadParams.Key,
      ContentType: uploadParams.ContentType,
      BodySize: uploadParams.Body.length
    });

    // Retry logic for upload
    let lastError;
    const maxRetries = 3;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Upload attempt ${attempt}/${maxRetries}`);
        
        // Use upload with progress tracking
        const upload = s3.upload(uploadParams);
        
        // Optional: Add progress tracking
        upload.on('httpUploadProgress', (progress) => {
          console.log(`Upload progress: ${Math.round(progress.loaded / progress.total * 100)}%`);
        });

        const result = await upload.promise();
        
        // Return the public URL
        const publicUrl = `${PUBLIC_URL}/${fileName}`;
        
        console.log('Upload successful:', { 
          publicUrl,
          etag: result.ETag,
          location: result.Location
        });
        
        return publicUrl;

      } catch (error) {
        lastError = error;
        console.error(`Upload attempt ${attempt} failed:`, {
          code: error.code,
          message: error.message,
          statusCode: error.statusCode,
          region: error.region
        });
        
        if (attempt < maxRetries) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          console.log(`Retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    // If all retries failed
    throw new Error(`Failed to upload image after ${maxRetries} attempts: ${lastError.message}`);

  } catch (error) {
    console.error('Cloudflare R2 upload error:', error);
    throw error;
  }
};

// Alternative upload method using putObject instead of upload
exports.uploadImageToCloudflareAlt = async (file, folder = 'images') => {
  try {
    // Validate the file
    validateImageFile(file);

    // Generate unique filename
    const fileName = generateFileName(file.originalname, folder);

    // Upload parameters for putObject
    const putParams = {
      Bucket: BUCKET_NAME,
      Key: fileName,
      Body: file.buffer,
      ContentType: file.mimetype,
      CacheControl: 'max-age=31536000',
      ContentLength: file.buffer.length,
      Metadata: {
        originalName: file.originalname,
        uploadedAt: new Date().toISOString(),
      }
    };

    console.log('Alternative upload with putObject...');
    
    // Use putObject instead of upload
    const result = await s3.putObject(putParams).promise();

    // Return the public URL
    const publicUrl = `${PUBLIC_URL}/${fileName}`;
    
    console.log('Alternative upload successful:', { 
      publicUrl,
      etag: result.ETag
    });
    
    return publicUrl;

  } catch (error) {
    console.error('Alternative upload error:', error);
    throw error;
  }
};

// Delete image from Cloudflare R2
exports.deleteImageFromCloudflare = async (imageUrl) => {
  try {
    // Extract the key from the URL
    const url = new URL(imageUrl);
    const key = url.pathname.substring(1); // Remove leading slash

    const deleteParams = {
      Bucket: BUCKET_NAME,
      Key: key
    };

    await s3.deleteObject(deleteParams).promise();
    console.log(`Successfully deleted image: ${key}`);

  } catch (error) {
    console.error('Cloudflare R2 delete error:', error);
    // Don't throw error for delete operations
  }
};

// Upload multiple images with better error handling
exports.uploadMultipleImages = async (files, folder = 'images') => {
  try {
    if (!files || !Array.isArray(files) || files.length === 0) {
      return [];
    }

    if (files.length > 4) {
      throw new Error('Maximum 4 images allowed');
    }

    console.log(`Uploading ${files.length} images...`);

    // Upload files sequentially to avoid overwhelming the connection
    const urls = [];
    for (let i = 0; i < files.length; i++) {
      try {
        console.log(`Uploading image ${i + 1}/${files.length}`);
        const url = await exports.uploadImageToCloudflare(files[i], folder);
        urls.push(url);
        
        // Small delay between uploads
        if (i < files.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      } catch (error) {
        console.error(`Failed to upload image ${i + 1}:`, error);
        // Try alternative method
        try {
          const url = await exports.uploadImageToCloudflareAlt(files[i], folder);
          urls.push(url);
        } catch (altError) {
          console.error(`Alternative upload also failed for image ${i + 1}:`, altError);
          throw error; // Re-throw original error
        }
      }
    }

    console.log(`Successfully uploaded ${urls.length} images`);
    return urls;

  } catch (error) {
    console.error('Multiple images upload error:', error);
    throw error;
  }
};

// Delete multiple images
exports.deleteMultipleImages = async (imageUrls) => {
  try {
    if (!imageUrls || !Array.isArray(imageUrls) || imageUrls.length === 0) {
      return;
    }

    // Delete all images concurrently
    const deletePromises = imageUrls.map(url => exports.deleteImageFromCloudflare(url));
    await Promise.all(deletePromises);

  } catch (error) {
    console.error('Multiple images delete error:', error);
    // Don't throw error for delete operations
  }
};

// Health check function
exports.healthCheck = async () => {
  try {
    return await testConnection();
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};

module.exports = {
  uploadImageToCloudflare: exports.uploadImageToCloudflare,
  uploadImageToCloudflareAlt: exports.uploadImageToCloudflareAlt,
  deleteImageFromCloudflare: exports.deleteImageFromCloudflare,
  uploadMultipleImages: exports.uploadMultipleImages,
  deleteMultipleImages: exports.deleteMultipleImages,
  healthCheck: exports.healthCheck
};