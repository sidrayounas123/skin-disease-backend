# 🚀 Easiest Railway Deployment Setup

## Step 1: Upload Models to Google Drive (2 minutes)

1. Go to [Google Drive](https://drive.google.com)
2. Click "New" → "File upload"
3. Upload both files:
   - `model1.pth` (329MB)
   - `model2.pth` (329MB)
4. Right-click each file → "Share" → "Get link"
5. Convert links to download format:
   - Change `.../view?usp=sharing` to `.../uc?export=download`

## Step 2: Update railway.py (1 minute)

Replace this section in `railway.py`:

```python
model_urls = {
    "model1.pth": "YOUR_MODEL1_DOWNLOAD_LINK",
    "model2.pth": "YOUR_MODEL2_DOWNLOAD_LINK"
}
```

Example:
```python
model_urls = {
    "model1.pth": "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID",
    "model2.pth": "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
}
```

## Step 3: Deploy to Railway (3 minutes)

1. Go to [Railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `skin-disease-backend` repository
4. Railway will automatically detect `railway.toml`
5. Set environment variables:
   - `FIREBASE_CREDENTIALS`: Your Firebase JSON
   - `FIREBASE_WEB_API_KEY`: `AIzaSyCqC7fkcWhLLdMlb080P7vQHwe3t-1YSSs`
6. Click "Deploy"

## Step 4: Update railway.toml

Change the start command:
```toml
[deploy]
startCommand = "python railway.py"
```

## ✅ Done!

Your app will:
- Deploy within Railway's size limits
- Download models automatically on first start
- Work exactly like before
- Be accessible at `https://your-app-name.up.railway.app`

**Total time: 10 minutes**
