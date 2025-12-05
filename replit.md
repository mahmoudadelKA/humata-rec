# Multi-Tool Arabic Platform (منصة الأدوات الذكية)

## Overview
A comprehensive web application with ten powerful tools, featuring a modern Arabic RTL interface with glassmorphism design.

## Tools

### 1. Speech-to-Text Live (تحويل الصوت لنص مباشر)
- Uses browser's native Web Speech API (webkitSpeechRecognition)
- Supports Arabic language recognition
- Real-time transcription display
- **Two recording modes:**
  - **Normal Recording:** Single-shot recording with manual stop
  - **Continuous Listening:** Long-running recording (up to 2 hours) that automatically restarts on errors, captures audio from external devices near the microphone
- Wake lock support to prevent screen sleep during continuous recording
- Copy-to-clipboard, download, and clear text functionality

### 2. YouTube Video Cutter (قص فيديو يوتيوب)
- Extract video information using yt-dlp
- Cut specific segments using ffmpeg
- Download clips as MP4 files
- Time format: MM:SS or HH:MM:SS
- Extended timeout (300s) for long videos

### 3. Anime Detector (كاشف الأنمي) - Dual Mode
- **Search by Image Tab:**
  - Uses trace.moe API for anime identification
  - STRICT similarity threshold: Only shows results if similarity >= 89%
  - If below threshold, displays: "No exact match found. Please use a clearer image."
  - Fetches actual anime name from Anilist GraphQL API
  - Video preview of the scene
- **Search by Name Tab:**
  - Users can type anime name (English/Japanese)
  - Fetches top 3 results from Anilist API
  - Displays covers, descriptions, episode count, genres, and scores

### 4. Podcast Detector (كاشف البودكاست) - UPGRADED with AI
- **From Image Mode (Screenshot/Cover):**
  - Uses Tesseract OCR (configured for English/Arabic) to extract text
  - Falls back to Gemini AI Vision for face/studio/logo recognition
  - Searches iTunes API with extracted text
  - Uses fuzzywuzzy for name similarity matching
  - Only displays results if Name Similarity >= 90%
  - Shows podcast artwork, artist, genre, and episode count
  - **Smart Links:** YouTube, Spotify, SoundCloud buttons for quick access
- **From Audio Clip Mode:**
  - Uses SpeechRecognition (Google Web Speech API) to transcribe audio
  - **Gemini AI Integration:** Identifies podcast name from transcript content
  - Processes first 60 seconds of audio
  - Searches iTunes API with AI-identified or keyword-based terms
  - **Smart Links:** YouTube, Spotify, SoundCloud buttons for each result

### 5. Audio File Transcription (تحويل ملف صوتي)
- Upload audio files (MP3, WAV, OGG, M4A, FLAC, AAC, WMA)
- Uses SpeechRecognition library with Google Speech API
- Converts audio to text (Arabic support)
- Download and copy extracted text

### 6. OCR - Extract Text from Image (استخراج النص من الصورة)
- Upload images (PNG, JPG, JPEG, GIF, WEBP)
- Uses Tesseract OCR for text extraction
- Supports Arabic and English languages
- Download and copy extracted text

### 7. Universal Media Downloader (تحميل فيديو/صوت من أي موقع) - UPDATED
- Download videos and audio from multiple platforms (YouTube, TikTok, Facebook, Instagram, Twitter, etc.)
- Uses yt-dlp library for universal media extraction
- **Two download options:**
  - **Video (MP4):** Automatically converts to MP4 format with high quality
  - **Audio (MP3):** Extracts audio and converts to MP3 at 192kbps quality
- Maximum duration: 1 hour
- Maximum file size: 500MB
- Files are deleted immediately after download to save server space
- Error handling for private content, unsupported sites, and large files

### 8. Smart PDF to Word Converter (محول PDF الذكي)
- Convert PDF files to Word (.docx) format
- Uses pdf2docx for layout-preserving conversion
- **Three conversion modes:**
  - **Original Language:** Converts PDF to Word without translation
  - **Translate to Arabic:** Converts and translates content to Arabic
  - **Translate to English:** Converts and translates content to English
- Uses deep_translator (Google Translate API) for translation
- Translates both paragraphs and table content
- Automatic cleanup of temporary files after download

### 9. Video to Text Transcription (تحويل الفيديو إلى نص) - UPGRADED
- **Two input modes:**
  - **File Upload:** Upload video files directly (MP4, MOV, AVI, MKV, WEBM, FLV, WMV, M4V)
  - **URL Input:** Paste video URLs from YouTube, TikTok, Facebook, Instagram, Twitter, etc.
- **Multi-language support** with language selector dropdown:
  - Arabic (default), English, French, Spanish, German, Turkish, Urdu, Hindi, Indonesian, Portuguese, Russian, Japanese, Korean, Chinese
  - Auto-detect option for mixed-language content
- Extracts audio from video using ffmpeg
- Compresses audio to 32kbps MP3 for efficient upload on slow connections
- Uses Gemini AI for accurate transcription
- **Supports videos up to 2 hours** (22,000 word limit for full-length episodes)
- Optimized for slow internet connections (0.72 Mbps)
- Copy, download, and clear transcribed text functionality
- Extended timeout (2 hours) for large file processing

### 10. AI Document Formatter (تنسيق ذكي للمستندات) - UPGRADED
- **AI-Powered Formatting:** Uses Gemini AI to understand and apply formatting
- **Supports both Word (.docx) and PDF files**
- **Two ways to specify formatting:**
  - **Text Description:** Describe the formatting you want in Arabic or English (e.g., "تنسيق بحث جامعي بخط Arial حجم 12")
  - **Reference File:** Upload an image, Word, or PDF file as a formatting reference
- **Smart formatting extraction:** AI analyzes reference files to understand font, margins, spacing, alignment
- **Features:**
  - Automatic PDF to DOCX conversion for editing
  - Heading detection and styling
  - First line indent support
  - Custom margins and line spacing
- **Modern dual-progress UI:** Shows both upload progress and AI processing progress
- Downloads formatted file as .docx
- Uses python-docx library with Gemini AI integration

## Tech Stack
- **Backend**: Python Flask
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Libraries**: yt-dlp, ffmpeg, SpeechRecognition, pytesseract, pydub, Pillow, fuzzywuzzy, python-Levenshtein, pdf2docx, python-docx, deep_translator, google-generativeai
- **APIs**: trace.moe, Anilist GraphQL, iTunes Search API, Google Gemini AI (Vision + Text)
- **AI Features**: Multi-key rotation (6 Gemini API keys), automatic failover on quota limits
- **Design**: Glassmorphism with RTL Arabic support

## API Routes
- `GET /` - Main page (renders index.html)
- `GET /logo.png` - Serve logo image
- `POST /video-info` - Get YouTube video information
- `POST /process-video` - Download and cut video segment
- `POST /search-anime` - Search anime from uploaded image (89% threshold)
- `POST /search-anime-by-name` - Search anime by name (Anilist API)
- `POST /search-podcast-by-image` - Search podcast from image OCR (90% threshold)
- `POST /search-podcast-by-audio` - Search podcast from audio transcription
- `POST /transcribe-file` - Transcribe audio file to text
- `POST /ocr-image` - Extract text from image using OCR
- `POST /download-video` - Universal media downloader (accepts `format` param: 'video' for MP4, 'audio' for MP3)
- `POST /convert-pdf` - PDF to Word converter (accepts `mode` param: 'original', 'ar', or 'en' for translation)
- `POST /transcribe-video` - Video to text transcription (accepts `input_type`: 'file' or 'url', video file or url)
- `POST /format-docx` - DOCX formatter (applies basic academic formatting and returns formatted file)
- `POST /format-document-ai` - AI-powered document formatter (accepts file, instructions, reference_file)

## Running the Application
The server runs on port 5000 using gunicorn with auto-reload and 1800s timeout (30 minutes for large videos).

## Project Structure
```
├── main.py             # Flask backend with all API routes
├── requirements.txt    # Python dependencies
├── logo.png            # Application logo
├── templates/
│   └── index.html      # Main frontend (SPA with all tools)
└── replit.md           # Project documentation
```

## Dependencies
- Flask
- yt-dlp
- requests
- gunicorn
- Pillow
- pydub
- pytesseract
- SpeechRecognition
- fuzzywuzzy
- python-Levenshtein
- pdf2docx
- python-docx
- deep_translator
- ffmpeg (system)
- tesseract (system)

## Recent Changes
- December 2024: Enhanced Continuous Listening feature with wake lock support and automatic restart on errors
- December 2024: Improved video downloader with better FFmpeg H.264/AAC encoding and faststart flag
- December 2024: Enhanced OCR with improved Tesseract config (--oem 3 --psm 6) for longer text support
- December 2024: Added Smart PDF to Word Converter with translation options (Arabic/English)
- December 2024: Added MP3 audio download support to Universal Downloader (192kbps quality)
- December 2024: Increased gunicorn worker timeout to 300 seconds for large file transfers
- December 2024: Added Podcast Detector with Image OCR and Audio modes
- December 2024: Upgraded Anime Detector with dual mode (Image + Name search)
- December 2024: Set strict similarity thresholds (Anime: 89%, Podcast: 90%)
- December 2025: **Gemini AI Integration** - Shared AI engine with multi-key rotation (6 keys)
- December 2025: **Podcast Detector Upgrade** - AI-powered identification from audio transcripts
- December 2025: **Smart Links** - YouTube, Spotify, SoundCloud buttons for all podcast results
- December 2025: **Continuous Listening** - Improved with Egyptian Arabic (ar-EG) dialect support
- December 2025: **Audio Transcription Fix** - Reduced chunk size to 30s, max duration 180s, better timeout handling
- December 2025: **Continuous Listening Sensitivity** - Disabled echo cancellation for external audio, increased alternatives
- December 2025: **AI-First Detection** - Anime/Podcast detectors now use Gemini AI Vision as primary method
- December 2025: **Enhanced Anime Detector** - Improved Gemini prompt with structured analysis of visual elements, added search links to Anilist, MyAnimeList, and Crunchyroll
- December 2025: **Enhanced Podcast Detector** - Improved Gemini prompt with structured format (PODCAST_NAME|HOST_NAMES|PLATFORM), returns host names and platform hints
- December 2025: **Long Audio Transcription** - Now supports files up to 3 hours (10800 seconds) using Gemini file upload API for transcription
- December 2025: **Improved Continuous Listening UI** - Added clear instructions explaining the feature uses device microphone only, with tips for best results
- December 2025: **Video to Text Transcription** - NEW tool for converting videos to text using Gemini AI, supports file uploads and URLs from multiple platforms (YouTube, TikTok, etc.)
- December 2025: **Multi-language Video Transcription** - Added language selector with 14 languages (Arabic default), auto-detect option, and increased word limit to 22,000 for 2-hour videos
- December 2025: **Extended API Key Support** - Now supports up to 12 Gemini API keys (GEMINI_API_KEY + GEMINI_KEY_1 through GEMINI_KEY_11)
- December 2025: **Smart Key Cooldown** - 1-hour cooldown for exhausted keys, prevents quota errors from repeated retries
- December 2025: **Mobile Responsive Design** - Added floating menu toggle button for mobile screens, improved sidebar UX
- December 2025: **GitHub Ready** - Added comprehensive README.md, cleaned up requirements.txt, updated .gitignore
- December 2025: **DOCX Formatter** - NEW tool for academic Word document formatting (2.5cm margins, Times New Roman 14pt, 1.5 line spacing, justified alignment)

## Required API Keys
To enable Gemini AI features (required for Video to Text, PDF OCR, and AI detection), add one or more of these secrets in the Replit Secrets tab:
- `GEMINI_API_KEY` - Primary Gemini API key (REQUIRED for video transcription)
- `GEMINI_KEY_1` through `GEMINI_KEY_11` - Additional keys for rotation (optional, recommended for heavy usage)

## API Key Rotation System
The application supports up to 12 Gemini API keys with intelligent load balancing:
- Automatic rotation across all available keys
- 1-hour cooldown period for exhausted keys (quota errors)
- Smart fallback when all keys are cooling down with time remaining display
- Keys are tracked independently and never reused before cooldown expires

**Note:** Without GEMINI_API_KEY, the video-to-text feature will show an error message asking you to configure the API key.

## Setup Instructions for New Imports
When importing this project to a new Replit account:
1. The project is pre-configured to run on Replit with all dependencies installed
2. Go to the "Secrets" tab in Replit
3. Add `GEMINI_API_KEY` with your Google Gemini API key (get one free from https://aistudio.google.com/app/apikey)
4. Optionally add `GEMINI_KEY_1` through `GEMINI_KEY_11` for load balancing
5. Optionally add `SESSION_SECRET` with any random string for session security
6. Optionally add `COOKIE_CONTENT` if you need YouTube cookie authentication for private videos
7. The server will automatically restart and detect your keys

## Replit Environment
- **System Dependencies**: FFmpeg and Tesseract OCR are pre-installed
- **Python Version**: 3.11
- **Workflow**: Automatically starts on port 5000 with gunicorn
- **Deployment**: Configured for Replit autoscale deployment

## Server Configuration
- Timeout: 7200 seconds (2 hours) to support long video transcription
- Workers: 2 gunicorn workers for better concurrency
- Port: 5000 (required for Replit hosting)
