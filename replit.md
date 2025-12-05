# Multi-Tool Arabic Platform (منصة الأدوات الذكية)

## Overview
The Multi-Tool Arabic Platform is a comprehensive web application offering ten powerful tools. Its primary purpose is to provide a suite of utilities for media processing, AI-driven content analysis, and document manipulation, all within a modern, Arabic RTL, glassmorphism-designed interface. The platform aims to serve users with efficient and intelligent tools for tasks like speech-to-text conversion, video cutting, anime and podcast detection, universal media downloading, PDF to Word conversion, video transcription, and AI-powered document formatting. It leverages advanced AI models and various APIs to deliver robust functionality, catering to both general users and those requiring specialized Arabic language support.

## Recent Changes (December 5, 2025)
- **Replit Environment Setup:** Successfully imported and configured the project for Replit deployment
- **System Dependencies:** Installed FFmpeg and Tesseract OCR via Nix packages
- **Python Dependencies:** Installed all required packages from requirements.txt
- **Workflow Configuration:** Configured Gunicorn to run on 0.0.0.0:5000 with webview output
- **Deployment Configuration:** Set up autoscale deployment with proper timeout settings (1800s)
- **Authentication Removed:** Replaced login/registration with simple "Enter AI World" button for direct access
- **Global Video Background:** YouTube video now plays as background throughout the entire site
- **Sidebar Navigation Fixes (New):**
  - Fixed toggle button responsiveness by keeping inline `onclick` attribute as fallback
  - Removed DOM cloning that was causing event listener issues
  - Added capture phase event listeners for better event handling
  - Improved mobile menu state management
  - Added localStorage persistence for active tool selection
  - Menu now properly opens/closes on click/touch
  - Sidebar stays open when clicking links on desktop
  - Sidebar closes after link click only on mobile view
- **Status:** Application is fully functional with responsive sidebar navigation

## User Preferences
I prefer simple language and clear explanations. I want iterative development where I can provide feedback at each stage. Ask before making major changes to the project's architecture or core functionalities. Ensure the application maintains its Arabic RTL design and glassmorphism aesthetic. All new features should seamlessly integrate with the existing UI/UX. Do not make changes to the `replit.md` file without explicit instruction.

## System Architecture
The platform is built on a Python Flask backend with a responsive frontend using HTML5, Tailwind CSS, and JavaScript.

**UI/UX Decisions:**
- **Design:** Modern glassmorphism aesthetic with a focus on Arabic Right-to-Left (RTL) layout.
- **Responsiveness:** Mobile-first design with a floating menu toggle for small screens.
- **Authentication:** Supabase integration for user login/registration and session management, blocking tool access until authentication.

**Technical Implementations & Feature Specifications:**
- **Speech-to-Text Live:** Utilizes `webkitSpeechRecognition` for real-time Arabic transcription, including continuous listening with wake lock and automatic restarts.
- **YouTube Video Cutter:** Extracts and cuts video segments using `yt-dlp` and `ffmpeg`.
- **Anime Detector:** Dual mode (Image/Name search) leveraging `trace.moe` and Anilist GraphQL API, with Gemini AI Vision for enhanced image analysis and strict similarity thresholds.
- **Podcast Detector:** Dual mode (Image/Audio) using Tesseract OCR, SpeechRecognition, and Gemini AI for identification, featuring smart links to streaming platforms and advanced prompt engineering for host/platform hints.
- **Audio File Transcription:** Transcribes audio files (various formats) to text using Google Speech API.
- **OCR - Extract Text from Image:** Extracts text from images (PNG, JPG, etc.) using Tesseract, supporting Arabic and English.
- **Universal Media Downloader:** Downloads video/audio from multiple platforms via `yt-dlp`, converting to MP4/MP3, with size and duration limits.
- **Smart PDF to Word Converter:** Converts PDFs to DOCX using `pdf2docx`, offering translation to Arabic or English via `deep_translator`.
- **Video to Text Transcription:** Transcribes video content (file upload or URL) to text using Gemini AI, supporting multiple languages, and optimized for large files and slow connections.
- **AI Document Formatter:** Formats Word/PDF documents using Gemini AI, allowing formatting specification via text description or reference files, and supporting features like heading styling, margins, and line spacing.

**System Design Choices:**
- **AI Integration:** Centralized Gemini AI engine with multi-key rotation (up to 12 keys), automatic failover, and smart cooldown for exhausted keys to prevent quota limits.
- **Media Processing:** Extensive use of `ffmpeg` for audio/video manipulation and `yt-dlp` for media extraction.
- **Backend Infrastructure:** Flask application served by Gunicorn with extended timeouts (1800s) for long-running tasks, configured for Replit deployment.
- **Concurrency:** 2 Gunicorn workers with 4 threads each for improved concurrency.
- **File Handling:** Temporary files are managed and deleted post-processing to conserve server space.
- **Proxy Configuration:** Flask app uses ProxyFix middleware to properly handle Replit's proxy setup.

## External Dependencies
- **Backend Framework:** Flask
- **Frontend Technologies:** HTML5, Tailwind CSS, JavaScript
- **Core Libraries:** `yt-dlp`, `ffmpeg` (system-level), `SpeechRecognition`, `pytesseract`, `pydub`, `Pillow`, `fuzzywuzzy`, `python-Levenshtein`, `pdf2docx`, `python-docx`, `deep_translator`, `google-generativeai`
- **APIs & Services:**
    - `trace.moe` (Anime detection)
    - `Anilist GraphQL API` (Anime information)
    - `iTunes Search API` (Podcast detection)
    - `Google Gemini AI` (Vision + Text - for advanced detection, transcription, and formatting)
    - `Supabase Auth` (User authentication and management)
- **OCR Engine:** `Tesseract` (system-level)

## Environment Variables
**Required:**
- `SESSION_SECRET` - Flask session secret (already configured in Replit)
- `GEMINI_API_KEY` or `GEMINI_KEY_1` - At least one Google Gemini API key for AI features

**Optional (for additional features):**
- `GEMINI_KEY_2` through `GEMINI_KEY_11` - Additional API keys for load balancing
- `SUPABASE_URL` - Supabase project URL for authentication
- `SUPABASE_ANON_KEY` - Supabase anonymous key for authentication
- `COOKIE_CONTENT` - YouTube cookies for downloading age-restricted content

**Notes:**
- The app supports up to 12 Gemini API keys for load balancing
- Without Supabase credentials, app runs in development mode (authentication disabled)
- Get Gemini API keys from: https://aistudio.google.com/app/apikey