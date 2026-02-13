#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "Installing system dependencies..."
# Render doesn't allow sudo apt-get on free tier easily within the build command sometimes,
# but we can try to use a cleaner approach or rely on Render's native environment if possible.
# However, for ffmpeg and tesseract, we usually need them.

pip install --upgrade pip
pip install -r requirements.txt

# If you are on a custom Docker image it would be better, but for 'env: python':
# We'll hope the environment has basic tools or we'll need to use a Dockerfile approach if this fails.
# But "argument list too long" is often due to the inline command length in render.yaml.
