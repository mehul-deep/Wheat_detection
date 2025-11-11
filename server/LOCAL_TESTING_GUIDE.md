# Local Testing Guide

## Quick Start (2 Options)

### Option 1: Full Test with Firebase (Recommended)

This tests the complete authentication flow locally.

#### Step 1: Authenticate with Google Cloud

Open a **new terminal** and run:

```bash
gcloud auth application-default login
```

This will:
1. Open your browser
2. Ask you to login with your Google account
3. Grant permissions to gcloud
4. Save credentials locally

#### Step 2: Run the App

```bash
cd /mnt/d/Duplicate/Wheat_detection/server
./run_local.sh
```

#### Step 3: Test

1. Open browser: http://localhost:5000
2. You'll be redirected to `/login`
3. Click "Sign Up" and create a test account
4. After signup, you'll be logged in automatically
5. Upload a wheat image to test detection
6. Check Firestore in Firebase Console to see your upload data

---

### Option 2: Quick Test (No Authentication)

This just checks if the app starts without testing authentication.

```bash
cd /mnt/d/Duplicate/Wheat_detection/server
./run_local_no_auth.sh
```

**Note:** This will show Firebase initialization errors, which is expected. Use this only to verify the app starts.

---

## Detailed Steps for Full Testing

### 1. Authenticate with gcloud (One-time setup)

```bash
# Make sure you're in the correct project
gcloud config set project wheat-detection-prod

# Authenticate
gcloud auth application-default login
```

Follow the prompts in your browser.

### 2. Start the Application

```bash
cd /mnt/d/Duplicate/Wheat_detection/server
./run_local.sh
```

You should see:
```
==================================
Starting Wheat Detection App (Local Mode)
==================================
Storage: Local filesystem
Models: model.pth, best.pt
Firebase: wheat-detection-cb988
URL: http://localhost:5000
==================================

Loading model: model.pth
Firebase Admin initialized with default credentials
Firestore client initialized
Model loaded and ready on cpu
 * Serving Flask app 'app'
 * Running on http://0.0.0.0:5000
```

### 3. Test the Authentication Flow

#### A. Create an Account
1. Open: http://localhost:5000
2. You'll be redirected to http://localhost:5000/login
3. Click the "Sign Up" tab
4. Fill in:
   - Full Name: Test User
   - Email: test@example.com
   - Password: test123 (min 6 characters)
   - Confirm Password: test123
5. Click "Create Account"

#### B. Verify Signup
- Should automatically log you in
- Should redirect to main page (/)
- Should show your name/email in the top bar
- Should show empty history sidebar

#### C. Test Upload
1. Click "Choose an image or drag it here"
2. Select a wheat image (JPG/PNG)
3. Click "Predict"
4. Wait for processing (loading overlay will show)
5. View results page with:
   - Original image
   - Overlay
   - Color mask
   - YOLO detections (if model is available)
   - Detection table

#### D. Verify History
1. Go back to home page (click logo or "Upload another image")
2. Check "Your History" sidebar
3. Should show your recent upload
4. Click on the history item to view previous results

#### E. Test Logout
1. Click "Logout" button in top bar
2. Should redirect to /login
3. Try accessing / directly - should redirect to /login

#### F. Test Login
1. On login page, enter your credentials
2. Click "Log In"
3. Should log you in and redirect to /

### 4. Verify Firestore Data

#### In Firebase Console:
1. Go to https://console.firebase.google.com
2. Select project: wheat-detection-cb988
3. Go to Firestore Database
4. Check collections:
   - `users/{your-user-id}`: Your user profile
   - `uploads/{upload-id}`: Your upload metadata

### 5. Check Local File Storage

```bash
# Check uploads
ls -la /mnt/d/Duplicate/Wheat_detection/server/uploads

# Check results
ls -la /mnt/d/Duplicate/Wheat_detection/server/static/results
```

You should see directories with UUIDs containing your uploaded images and results.

---

## Troubleshooting

### Issue: "Firebase Admin not initialized"
**Solution:**
```bash
# Re-authenticate
gcloud auth application-default login

# Verify authentication
gcloud auth application-default print-access-token
```

### Issue: "ModuleNotFoundError: No module named 'firebase_admin'"
**Solution:**
```bash
source .venv/bin/activate
pip install firebase-admin google-cloud-firestore
```

### Issue: Port 5000 already in use
**Solution:**
```bash
# Find process using port 5000
lsof -ti:5000

# Kill it
kill -9 $(lsof -ti:5000)

# Or use a different port
export PORT=5001
python app.py
```

### Issue: "Firestore permission denied"
**Solution:**
- Make sure you deployed Firestore rules (see main DEPLOY_AUTH.md)
- Or temporarily set Firestore to test mode in Firebase Console

### Issue: Models not found
**Solution:**
```bash
# Make sure you're in the server directory
cd /mnt/d/Duplicate/Wheat_detection/server

# Check if models exist
ls -la model.pth best.pt
```

### Issue: Can't access /login
**Solution:**
- Check app.py for syntax errors
- Check console output for Python errors
- Verify templates/login.html exists

---

## Expected Behavior

### ✅ Success Indicators:
- [ ] App starts without errors
- [ ] Can access http://localhost:5000/login
- [ ] Can create new account
- [ ] Automatically logged in after signup
- [ ] User info displays in topbar
- [ ] Can upload image and see results
- [ ] Upload appears in history sidebar
- [ ] Can click history item to view previous results
- [ ] Can logout successfully
- [ ] Can login with existing credentials
- [ ] Firestore shows user and upload data

### ⚠️  Expected Warnings (Safe to Ignore):
- "Compute Engine Metadata server unavailable" - Normal for local development
- YOLO model warnings if best.pt not found

### ❌ Error Indicators (Need to Fix):
- "Firebase Admin not initialized" - Need to run `gcloud auth`
- "Port already in use" - Kill existing process or use different port
- "ModuleNotFoundError" - Install missing dependencies
- Python syntax errors - Check code changes

---

## Next Steps After Local Testing

Once local testing is complete and everything works:

1. **Deploy Firestore Rules**:
   ```bash
   firebase deploy --only firestore:rules
   ```

2. **Deploy to Cloud Run**:
   ```bash
   cd /mnt/d/Duplicate/Wheat_detection/server

   # Build
   gcloud builds submit \
     --tag us-central1-docker.pkg.dev/wheat-detection-prod/wheat-detect/web:latest

   # Deploy
   gcloud run deploy wheat-web \
     --image us-central1-docker.pkg.dev/wheat-detection-prod/wheat-detect/web:latest \
     --region us-central1 \
     --platform managed \
     --allow-unauthenticated \
     --cpu 2 --memory 4Gi \
     --env-vars-file run-env.yaml
   ```

3. **Test Production**:
   - Visit: https://wheat-web-795033415293.us-central1.run.app
   - Test complete flow end-to-end

---

## Quick Command Reference

```bash
# Start with Firebase
./run_local.sh

# Start without Firebase (testing only)
./run_local_no_auth.sh

# Authenticate
gcloud auth application-default login

# View logs (while app is running)
# Check terminal output

# Stop server
# Press Ctrl+C in terminal

# Check Firestore
# Visit Firebase Console → Firestore Database
```
