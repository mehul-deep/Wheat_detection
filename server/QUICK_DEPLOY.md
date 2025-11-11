# Quick Deploy Guide - Firebase Auth Integration

## Prerequisites Checklist
- [x] Firebase project created: `wheat-detection-cb988`
- [x] Email/Password auth enabled in Firebase Console
- [x] Firestore database created in `us-central1`
- [x] Code updated with authentication

## One-Time Setup (Run These Once)

### 1. Install Firebase CLI
```bash
npm install -g firebase-tools
firebase login
```

### 2. Initialize Firebase in Project
```bash
cd /mnt/d/Duplicate/Wheat_detection/server
firebase init firestore
# Select: wheat-detection-cb988
# Use firestore.rules
```

### 3. Deploy Firestore Rules
```bash
firebase deploy --only firestore:rules
```

### 4. Grant Cloud Run Service Account Permissions
```bash
# Firebase Admin SDK access
gcloud projects add-iam-policy-binding wheat-detection-cb988 \
  --member="serviceAccount:795033415293-compute@developer.gserviceaccount.com" \
  --role="roles/firebase.sdkAdminServiceAgent"

# Firestore access
gcloud projects add-iam-policy-binding wheat-detection-cb988 \
  --member="serviceAccount:795033415293-compute@developer.gserviceaccount.com" \
  --role="roles/datastore.user"
```

## Deploy to Cloud Run (Run Every Time You Update Code)

```bash
cd /mnt/d/Duplicate/Wheat_detection/server

# Build container
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/wheat-detection-prod/wheat-detect/web:latest

# Deploy to Cloud Run
gcloud run deploy wheat-web \
  --image us-central1-docker.pkg.dev/wheat-detection-prod/wheat-detect/web:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --cpu 2 --memory 4Gi \
  --env-vars-file run-env.yaml
```

## Test After Deployment

1. Visit: https://wheat-web-795033415293.us-central1.run.app
2. Should redirect to `/login`
3. Click "Sign Up" and create test account
4. After signup, should auto-login and see main app
5. Upload an image
6. Check "Your History" sidebar shows the upload
7. Click logout button
8. Verify redirected back to login

## Troubleshooting Commands

```bash
# View logs
gcloud run services logs read wheat-web --region us-central1 --limit 50

# Check service status
gcloud run services describe wheat-web --region us-central1

# Check Firestore rules are deployed
firebase firestore:rules get

# List IAM permissions
gcloud projects get-iam-policy wheat-detection-cb988 \
  --flatten="bindings[].members" \
  --filter="bindings.members:795033415293-compute@developer.gserviceaccount.com"
```

## Quick Fixes

### "Firebase Admin not initialized"
```bash
# Verify service account has permissions
gcloud projects get-iam-policy wheat-detection-cb988 | grep 795033415293-compute
```

### "Permission denied" on Firestore
```bash
# Redeploy rules
cd /mnt/d/Duplicate/Wheat_detection/server
firebase deploy --only firestore:rules
```

### Users can't login
- Check Firebase Console → Authentication → Users
- Check browser console for JavaScript errors
- Verify Firebase config matches in both HTML files

### Old version still running
```bash
# Force new deployment
gcloud run deploy wheat-web \
  --image us-central1-docker.pkg.dev/wheat-detection-prod/wheat-detect/web:latest \
  --region us-central1 \
  --no-traffic

# Then update traffic to new revision
```

## Files Changed Summary

### New Files
- `templates/login.html` - Login/signup page
- `templates/index_auth.html` - Main app with dashboard
- `firestore.rules` - Security rules
- `DEPLOY_AUTH.md` - Full guide
- `AUTHENTICATION_SUMMARY.md` - Overview
- `QUICK_DEPLOY.md` - This file

### Modified Files
- `app.py` - Added auth routes and Firebase integration
- `requirements.txt` - Added firebase-admin, google-cloud-firestore
- `run-env.yaml` - Added FIREBASE_PROJECT_ID

## Support
- Detailed guide: See `DEPLOY_AUTH.md`
- Complete overview: See `AUTHENTICATION_SUMMARY.md`
