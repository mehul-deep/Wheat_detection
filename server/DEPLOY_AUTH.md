# Deployment Guide - Firebase Authentication Integration

## Overview
This guide covers deploying the Wheat Detection app with Firebase Authentication to Google Cloud Run.

## Prerequisites
- Firebase project: `wheat-detection-cb988`
- GCP project: `wheat-detection-prod`
- Firestore database enabled in `us-central1`
- Email/Password authentication enabled in Firebase

## Step 1: Deploy Firestore Security Rules

```bash
# Install Firebase CLI if not already installed
npm install -g firebase-tools

# Login to Firebase
firebase login

# Initialize Firebase in the project (if not done)
cd /mnt/d/Duplicate/Wheat_detection/server
firebase init firestore
# Select: wheat-detection-cb988
# Use firestore.rules as rules file
# Skip firestore.indexes.json

# Deploy security rules
firebase deploy --only firestore:rules
```

## Step 2: Grant Cloud Run Service Account Firebase Permissions

Your Cloud Run service needs permissions to verify Firebase tokens and access Firestore:

```bash
# Get your Cloud Run service account email
# It's usually: PROJECT_NUMBER-compute@developer.gserviceaccount.com
# Or check with:
gcloud run services describe wheat-web --region us-central1 --format='value(spec.template.spec.serviceAccountName)'

# Grant Firebase Admin SDK access
gcloud projects add-iam-policy-binding wheat-detection-cb988 \
  --member="serviceAccount:795033415293-compute@developer.gserviceaccount.com" \
  --role="roles/firebase.sdkAdminServiceAgent"

# Grant Firestore User access
gcloud projects add-iam-policy-binding wheat-detection-cb988 \
  --member="serviceAccount:795033415293-compute@developer.gserviceaccount.com" \
  --role="roles/datastore.user"
```

## Step 3: Build and Deploy to Cloud Run

```bash
cd /mnt/d/Duplicate/Wheat_detection/server

# Build the container image
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/wheat-detection-prod/wheat-detect/web:latest \
  --project wheat-detection-prod

# Deploy to Cloud Run with updated environment
gcloud run deploy wheat-web \
  --image us-central1-docker.pkg.dev/wheat-detection-prod/wheat-detect/web:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 4Gi \
  --env-vars-file run-env.yaml \
  --project wheat-detection-prod
```

## Step 4: Verify Deployment

1. **Test Login Page:**
   ```
   https://wheat-web-795033415293.us-central1.run.app/login
   ```

2. **Create a test account:**
   - Click "Sign Up" tab
   - Enter: Name, Email, Password
   - Submit

3. **Verify Firestore:**
   - Go to Firebase Console → Firestore Database
   - Check `users` collection has your user document
   - Check `uploads` collection (will populate after first upload)

4. **Test Upload:**
   - After login, upload a wheat image
   - Verify detection results
   - Check that upload appears in "Your History" sidebar
   - Verify Firestore `uploads` collection has the upload record

## Architecture

### Authentication Flow
1. User visits `/` → redirected to `/login`
2. User signs up/logs in with Firebase Auth (client-side)
3. Client gets Firebase ID token
4. Client sends token to `/auth/verify`
5. Server verifies token with Firebase Admin SDK
6. Server creates Flask session with user_id
7. User redirected to main app

### Data Storage
- **GCS Buckets:** Images stored in existing buckets (unchanged)
- **Firestore:**
  - `users/{userId}`: User profiles and stats
  - `uploads/{uploadId}`: Upload metadata and URLs

### Security
- All routes except `/login` require authentication
- Firestore rules enforce user can only access their own data
- Firebase Admin SDK verifies all ID tokens server-side
- Sessions use Flask's secure session management

## Troubleshooting

### "Firebase Admin not initialized"
- Check Cloud Run service account has correct roles
- Verify `FIREBASE_PROJECT_ID` env var is set correctly

### "Firestore permission denied"
- Deploy firestore.rules: `firebase deploy --only firestore:rules`
- Check service account has `roles/datastore.user`

### Users redirected to login after signup
- Check browser console for JavaScript errors
- Verify Firebase config in `login.html` and `index_auth.html` matches your project

### "Invalid token" errors
- Ensure system clocks are synchronized
- Check Firebase project ID matches in all configs
- Verify token is being sent in request

## Local Testing (Optional)

To test locally with Firebase:

1. **Create service account key:**
   ```bash
   gcloud iam service-accounts keys create firebase-key.json \
     --iam-account=795033415293-compute@developer.gserviceaccount.com \
     --project=wheat-detection-prod
   ```

2. **Set environment variable:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/firebase-key.json"
   export FIREBASE_PROJECT_ID="wheat-detection-cb988"
   ```

3. **Run locally:**
   ```bash
   cd /mnt/d/Duplicate/Wheat_detection/server
   python app.py
   ```

4. **Access at:** http://localhost:5000/login

## Monitoring

### View Logs
```bash
gcloud run services logs read wheat-web --region us-central1 --limit 50
```

### Check Authentication Events
Firebase Console → Authentication → Users
- View registered users
- Check sign-in timestamps

### Check Firestore Data
Firebase Console → Firestore Database
- `users` collection: User profiles
- `uploads` collection: Upload history

## Cost Considerations

- **Firestore:** Free tier includes 50K reads, 20K writes, 20K deletes per day
- **Firebase Auth:** Free for unlimited users
- **Cloud Run:** Unchanged (existing usage)

For typical usage (< 100 uploads/day), Firestore will remain in free tier.

## Next Steps

1. **Email Verification:** Add email verification on signup
2. **Password Reset:** Implement forgot password flow
3. **Social Login:** Add Google/GitHub OAuth
4. **Admin Dashboard:** Create admin role with analytics
5. **Rate Limiting:** Add per-user upload limits
6. **Batch Operations:** Allow users to download all their results
