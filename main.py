import os
import re
import uuid
import tempfile
import subprocess
import base64
import logging
from flask import Flask, render_template, request, jsonify, send_file, after_this_request
import yt_dlp
import requests
import speech_recognition as sr
from pydub import AudioSegment
import pytesseract
from PIL import Image
from fuzzywuzzy import fuzz
import google.generativeai as genai

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma'}

ANIME_SIMILARITY_THRESHOLD = 0.85
PODCAST_NAME_SIMILARITY_THRESHOLD = 90

GEMINI_KEYS = [
    os.environ.get("GEMINI_KEY_1"),
    os.environ.get("GEMINI_KEY_2"),
    os.environ.get("GEMINI_KEY_3"),
    os.environ.get("GEMINI_KEY_4"),
    os.environ.get("GEMINI_KEY_5"),
    os.environ.get("GEMINI_KEY_6"),
    os.environ.get("GEMINI_KEY_7"),
    os.environ.get("GEMINI_KEY_8"),
    os.environ.get("GEMINI_KEY_9"),
    os.environ.get("GEMINI_KEY_10"),
    os.environ.get("GEMINI_KEY_11"),
    os.environ.get("GEMINI_API_KEY"),
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

current_key_index = 0
key_usage_count = {}
exhausted_keys_tracker = {}


def get_balanced_key_index(allow_exhausted_fallback=False):
    """
    Get the next key index for load balancing across all keys with smart rotation.
    
    Args:
        allow_exhausted_fallback: If True, return earliest exhausted key when all are cooling down.
                                   If False, return None when all keys are cooling down.
    
    Returns:
        int: Key index to use, or None if no keys available (when allow_exhausted_fallback=False)
    """
    global current_key_index, key_usage_count, exhausted_keys_tracker
    import time

    if not GEMINI_KEYS:
        return None

    current_time = time.time()

    expired_keys = [
        k for k, v in exhausted_keys_tracker.items() if current_time - v > 3600
    ]
    for k in expired_keys:
        del exhausted_keys_tracker[k]
        logging.info(f"API key {k + 1} cooldown expired, now available")

    available_keys = [
        i for i in range(len(GEMINI_KEYS)) if i not in exhausted_keys_tracker
    ]

    if not available_keys:
        if exhausted_keys_tracker:
            earliest_key = min(exhausted_keys_tracker.items(),
                               key=lambda x: x[1])
            time_remaining = 3600 - (current_time - earliest_key[1])
            logging.warning(
                f"All API keys exhausted. Earliest key {earliest_key[0] + 1} available in {time_remaining/60:.1f} minutes"
            )

            if allow_exhausted_fallback:
                return earliest_key[0]
            else:
                return None
        else:
            return 0

    min_usage = min(key_usage_count.get(i, 0) for i in available_keys)
    least_used = [
        i for i in available_keys if key_usage_count.get(i, 0) == min_usage
    ]

    index = least_used[0]
    key_usage_count[index] = key_usage_count.get(index, 0) + 1

    return index


def get_time_until_key_available():
    """Get the time in seconds until the earliest exhausted key becomes available"""
    import time
    if not exhausted_keys_tracker:
        return 0
    current_time = time.time()
    earliest_key = min(exhausted_keys_tracker.items(), key=lambda x: x[1])
    time_remaining = 3600 - (current_time - earliest_key[1])
    return max(0, time_remaining)


def mark_key_exhausted(key_index):
    """Mark a key as exhausted with timestamp"""
    import time
    exhausted_keys_tracker[key_index] = time.time()
    logging.warning(
        f"API key {key_index + 1} marked as exhausted, will retry after 1 hour"
    )


def get_available_keys_count():
    """Get count of currently available (non-exhausted) keys"""
    import time
    current_time = time.time()
    available = len([
        k for k in range(len(GEMINI_KEYS))
        if k not in exhausted_keys_tracker or current_time -
        exhausted_keys_tracker[k] > 3600
    ])
    return available


logging.info(f"Loaded {len(GEMINI_KEYS)} Gemini API keys for load balancing")


def get_next_gemini_key(current_index=0):
    """Get the next available Gemini API key with rotation"""
    if not GEMINI_KEYS:
        return None, -1
    next_index = (current_index + 1) % len(GEMINI_KEYS)
    return GEMINI_KEYS[next_index], next_index


def call_gemini_text(prompt, max_retries=None):
    """
    Call Gemini API for text-only prompts with automatic key rotation on quota errors.
    Cycles through GEMINI_KEY_1 to GEMINI_KEY_5 on 429 errors.
    
    Args:
        prompt: The text prompt to send to the model
        max_retries: Maximum number of key rotations to try (defaults to number of available keys)
    
    Returns:
        str: Response text or None if all keys exhausted
    """
    if not GEMINI_KEYS:
        logging.warning("No Gemini API keys configured")
        return None

    if max_retries is None:
        max_retries = len(GEMINI_KEYS)

    for attempt in range(max_retries):
        key_index = attempt % len(GEMINI_KEYS)
        api_key = GEMINI_KEYS[key_index]

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            response = model.generate_content(prompt)

            if response and response.text:
                return response.text.strip()
            return None

        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'quota' in error_str or 'rate' in error_str or 'limit' in error_str:
                logging.warning(
                    f"Gemini text key {key_index + 1} quota exceeded, rotating to next key..."
                )
                continue
            else:
                logging.error(f"Gemini text API error: {e}")
                return None

    logging.error("All Gemini API keys exhausted for text")
    return None


def call_gemini_vision(image_or_path, prompt, max_retries=None):
    """
    Call Gemini Vision API with automatic key rotation on quota errors.
    Cycles through GEMINI_KEY_1 to GEMINI_KEY_5 on 429 errors.
    
    Args:
        image_or_path: Either a PIL Image object or path to image file
        prompt: The prompt to send to the model
        max_retries: Maximum number of key rotations to try (defaults to number of available keys)
    
    Returns:
        str: Response text or None if all keys exhausted
    """
    if not GEMINI_KEYS:
        logging.warning("No Gemini API keys configured")
        return None

    if max_retries is None:
        max_retries = len(GEMINI_KEYS)

    if isinstance(image_or_path, str):
        img = Image.open(image_or_path)
    else:
        img = image_or_path

    for attempt in range(max_retries):
        key_index = attempt % len(GEMINI_KEYS)
        api_key = GEMINI_KEYS[key_index]

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')

            response = model.generate_content([prompt, img])

            if response and response.text:
                return response.text.strip()
            return None

        except Exception as e:
            error_str = str(e).lower()
            if '429' in error_str or 'quota' in error_str or 'rate' in error_str or 'limit' in error_str:
                logging.warning(
                    f"Gemini key {key_index + 1} quota exceeded, rotating to next key..."
                )
                continue
            else:
                logging.error(f"Gemini API error: {e}")
                return None

    logging.error("All Gemini API keys exhausted")
    return None


def allowed_image(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def identify_podcast_with_gemini(image_path):
    """Use Gemini AI Vision to identify podcast from image"""
    prompt = """Analyze this image carefully. It could be a YouTube video screenshot, podcast artwork, or a photo of podcast hosts.
    
    Identify the Podcast Name, Channel Name, Show Name, or Host Names based on:
    - Visible logos, titles, or text
    - Recognizable faces of podcast hosts or celebrities
    - Studio setup, microphones, or recording environment
    - Channel branding or show graphics
    - Any visual elements that identify the content
    
    Return the response in this format: PODCAST_NAME|HOST_NAMES|PLATFORM
    - PODCAST_NAME: The main podcast/show/channel name
    - HOST_NAMES: Names of visible hosts (or 'Unknown' if not recognized)
    - PLATFORM: Where this podcast is likely found (YouTube, Spotify, Apple Podcasts, or General)
    
    If you cannot identify it at all, return 'UNKNOWN'.
    Be specific and accurate. Arabic and English names are both acceptable.
    
    Example: The Joe Rogan Experience|Joe Rogan|Spotify
    Example: بودكاست فنجان|عبدالرحمن أبومالح|YouTube"""

    result = call_gemini_vision(image_path, prompt)

    if result and result.upper() != 'UNKNOWN' and len(result) > 2:
        return result
    return None


def identify_podcast_from_transcript(transcript):
    """Use Gemini AI to identify podcast from transcribed audio text"""
    prompt = f"""Based on this audio transcript from a podcast or show, identify the Podcast name, Host, or Show name.
    
Transcript:
{transcript[:2000]}

Instructions:
- Look for any mentions of podcast names, show names, host introductions, channel names
- Consider common podcast phrases like "welcome to...", "this is...", "you're listening to..."
- Return ONLY the podcast/show name, nothing else
- If you cannot identify it, return 'UNKNOWN'
- Be specific and accurate."""

    result = call_gemini_text(prompt)

    if result and result.upper() != 'UNKNOWN' and len(result) > 2:
        clean_result = result.replace('"', '').replace("'", '').strip()
        if clean_result.upper() != 'UNKNOWN':
            return clean_result
    return None


def compress_audio_for_upload(audio_path):
    """Compress audio to low-bitrate MP3 for faster upload to Gemini"""
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        compressed_path = audio_path.rsplit('.', 1)[0] + '_compressed.mp3'
        audio.export(compressed_path, format="mp3", bitrate="32k")

        logging.info(
            f"Audio compressed for upload: {audio_path} -> {compressed_path}")
        return compressed_path
    except Exception as e:
        logging.error(f"Audio compression failed: {e}")
        return audio_path


LANGUAGE_NAMES = {
    'ar': 'Arabic',
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'tr': 'Turkish',
    'ur': 'Urdu',
    'hi': 'Hindi',
    'id': 'Indonesian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'auto': 'Auto-detect'
}


def transcribe_audio_with_gemini(audio_path, language='ar'):
    """
    Use Gemini to transcribe audio file - supports videos up to 2 hours.
    Features:
    - Smart load balancing across all 11 API keys
    - Automatic key rotation on quota exhaustion
    - 30-minute timeout for long videos (1800 seconds)
    - Multiple retry attempts per key with exponential backoff
    - Connection resilience for slow internet (< 1 Mbps upload)
    """
    import time

    if not GEMINI_KEYS:
        logging.warning("No Gemini API keys configured for transcription")
        raise ValueError(
            "لم يتم إعداد مفاتيح Gemini API. يرجى إضافة GEMINI_API_KEY في الإعدادات."
        )

    compressed_path = compress_audio_for_upload(audio_path)
    upload_path = compressed_path if compressed_path != audio_path else audio_path

    language_name = LANGUAGE_NAMES.get(language, 'Arabic')
    language_instruction = ""
    if language == 'auto':
        language_instruction = "Auto-detect the language spoken in the audio and transcribe in that same language."
    else:
        language_instruction = f"The audio is in {language_name}. Transcribe in {language_name} exactly as spoken - do NOT translate to any other language."

    prompt = f"""Transcribe this entire audio file completely and accurately. This is a FULL transcription request - transcribe EVERYTHING up to 22,000 words.

{language_instruction}

Instructions:
- Transcribe ALL speech in the audio from start to finish, word by word
- Do NOT skip any parts - transcribe the ENTIRE audio file completely
- Keep the original language - do NOT translate
- If Arabic, preserve Arabic diacritics if present
- Maintain paragraph breaks for different speakers or topics
- If there are multiple speakers, indicate speaker changes with new paragraphs
- Output ONLY the transcription text, no commentary, no timestamps, no summaries
- If the audio is unclear in parts, transcribe what you can hear
- For long audio (1-2+ hours), continue transcribing until the very end
- Maximum output: 22,000 words - use all of it if needed for 2-hour videos"""

    max_retries_per_key = 5
    base_retry_delay = 3
    last_error = None
    local_exhausted_keys = set()

    total_keys = len(GEMINI_KEYS)
    available_at_start = get_available_keys_count()
    logging.info(
        f"Starting transcription with {available_at_start}/{total_keys} keys available"
    )

    for rotation in range(total_keys * 2):
        key_index = get_balanced_key_index(allow_exhausted_fallback=False)

        if key_index is None:
            if not GEMINI_KEYS:
                logging.error("No API keys configured")
                raise ValueError(
                    "لم يتم إعداد مفاتيح Gemini API. يرجى إضافة GEMINI_API_KEY في الإعدادات."
                )
            else:
                time_until_available = get_time_until_key_available()
                minutes_remaining = time_until_available / 60
                logging.warning(
                    f"All keys in cooldown. Time until next available: {minutes_remaining:.1f} minutes"
                )
                raise ValueError(
                    f"تم استنفاد حصة جميع مفاتيح API. الرجاء الانتظار {int(minutes_remaining)} دقيقة أو إضافة مفاتيح جديدة."
                )

        if key_index in local_exhausted_keys:
            continue

        api_key = GEMINI_KEYS[key_index]

        for retry in range(max_retries_per_key):
            audio_file = None
            retry_delay = base_retry_delay * (2**retry)

            try:
                genai.configure(api_key=api_key)

                logging.info(
                    f"[Key {key_index + 1}/{total_keys}] Uploading audio file (attempt {retry + 1}/{max_retries_per_key})..."
                )

                audio_file = genai.upload_file(upload_path)

                model = genai.GenerativeModel('gemini-2.0-flash')

                logging.info(
                    f"[Key {key_index + 1}/{total_keys}] Starting transcription (timeout: 30 minutes)..."
                )

                response = model.generate_content(
                    [prompt, audio_file], request_options={"timeout": 1800})

                try:
                    genai.delete_file(audio_file.name)
                except:
                    pass

                if compressed_path != audio_path and os.path.exists(
                        compressed_path):
                    try:
                        os.remove(compressed_path)
                    except:
                        pass

                if response and response.text:
                    text = response.text.strip()
                    word_count = len(text.split())
                    logging.info(
                        f"[Key {key_index + 1}/{total_keys}] Transcription successful! {len(text)} chars, ~{word_count} words"
                    )
                    return text

                logging.warning(
                    f"[Key {key_index + 1}/{total_keys}] Empty response, retrying in {retry_delay}s..."
                )
                last_error = "Empty response from API"
                time.sleep(retry_delay)
                continue

            except Exception as e:
                error_str = str(e).lower()
                last_error = str(e)

                try:
                    if audio_file:
                        genai.delete_file(audio_file.name)
                except:
                    pass

                if '429' in error_str or 'quota' in error_str or 'rate' in error_str or 'limit' in error_str or 'exhausted' in error_str:
                    logging.warning(
                        f"[Key {key_index + 1}/{total_keys}] Quota exhausted, marking and switching..."
                    )
                    local_exhausted_keys.add(key_index)
                    mark_key_exhausted(key_index)
                    break

                elif 'timeout' in error_str or 'deadline' in error_str or 'timed out' in error_str:
                    logging.warning(
                        f"[Key {key_index + 1}/{total_keys}] Timeout on attempt {retry + 1}, retrying in {retry_delay * 2}s..."
                    )
                    time.sleep(retry_delay * 2)
                    continue

                elif 'connection' in error_str or 'network' in error_str or 'unavailable' in error_str or 'reset' in error_str or 'broken' in error_str:
                    logging.warning(
                        f"[Key {key_index + 1}/{total_keys}] Connection error: {str(e)[:100]}, retrying in {retry_delay * 3}s..."
                    )
                    time.sleep(retry_delay * 3)
                    continue

                elif 'invalid' in error_str or 'unauthorized' in error_str or 'permission' in error_str or 'api_key' in error_str:
                    logging.error(
                        f"[Key {key_index + 1}/{total_keys}] Invalid API key, skipping..."
                    )
                    local_exhausted_keys.add(key_index)
                    mark_key_exhausted(key_index)
                    break

                elif 'upload' in error_str or 'file' in error_str:
                    logging.warning(
                        f"[Key {key_index + 1}/{total_keys}] Upload error: {str(e)[:100]}, retrying in {retry_delay * 2}s..."
                    )
                    time.sleep(retry_delay * 2)
                    continue

                else:
                    logging.error(
                        f"[Key {key_index + 1}/{total_keys}] Error: {str(e)[:150]}, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    continue

        if len(local_exhausted_keys) >= total_keys:
            break

    if compressed_path != audio_path and os.path.exists(compressed_path):
        try:
            os.remove(compressed_path)
        except:
            pass

    available_now = get_available_keys_count()
    logging.error(
        f"Transcription failed. Keys exhausted in session: {len(local_exhausted_keys)}/{total_keys}. Available now: {available_now}. Last error: {last_error}"
    )

    if len(local_exhausted_keys) >= total_keys:
        raise ValueError(
            "تم استنفاد حصة جميع مفاتيح API. حاول مرة أخرى بعد ساعة أو أضف مفاتيح جديدة."
        )
    else:
        raise ValueError(
            f"فشل في تحويل الصوت إلى نص. الخطأ: {str(last_error)[:100]}")


def generate_podcast_search_links(podcast_name):
    """Generate search links for YouTube, Spotify, and SoundCloud"""
    if not podcast_name or not podcast_name.strip():
        return {'youtube': '', 'spotify': '', 'soundcloud': ''}
    from urllib.parse import quote
    clean_name = podcast_name.strip()
    encoded_name = quote(clean_name)
    return {
        'youtube':
        f"https://www.youtube.com/results?search_query={encoded_name}",
        'spotify': f"https://open.spotify.com/search/{encoded_name}",
        'soundcloud': f"https://soundcloud.com/search?q={encoded_name}"
    }


def identify_anime_with_gemini(image_path):
    """Use Gemini AI Vision to identify anime from image"""
    prompt = """Analyze this anime screenshot or image carefully.
    Identify the anime title based on the characters, art style, scene, character designs, backgrounds, or any recognizable elements.
    
    Instructions:
    - If you recognize the anime, return the title in this format: ANIME_NAME|EPISODE_INFO|DESCRIPTION
    - ANIME_NAME: The official English or Romaji title of the anime
    - EPISODE_INFO: Episode number if recognizable, or 'Unknown' if not
    - DESCRIPTION: Brief description of the scene or characters shown (1-2 sentences)
    - If you cannot identify it at all, return 'UNKNOWN'
    - Be as specific and accurate as possible
    - Consider popular anime series, movies, and OVAs
    
    Example response: Demon Slayer: Kimetsu no Yaiba|Episode 19|Tanjiro performing the Hinokami Kagura dance move against Rui"""

    result = call_gemini_vision(image_path, prompt)

    if result and result.upper() != 'UNKNOWN' and len(result) > 2:
        return result.replace('"', '').strip()
    return None


def identify_anime_by_description(description):
    """Use Gemini AI to identify anime from text description"""
    prompt = f"""أنت خبير في الأنمي. بناءً على هذا الوصف، حدد اسم الأنمي.

الوصف:
{description}

التعليمات:
- حلل الوصف للبحث عن أسماء الشخصيات، عناصر القصة، الإعدادات، القوى الخارقة، أو أي سمات مميزة
- ابحث عن أي تفاصيل يمكن أن تشير إلى أنمي معين مثل:
  * أسماء الشخصيات (بالعربية أو الإنجليزية أو اليابانية)
  * وصف المظهر أو الملابس المميزة
  * القوى الخاصة أو الأسلحة
  * الأحداث أو المشاهد المميزة
  * أسلوب الرسم أو الأنيميشن
  * أسماء الأماكن أو العوالم
- أعد الرد بهذا الشكل: ANIME_NAME|CONFIDENCE|ALTERNATIVES
- ANIME_NAME: الاسم الأكثر احتمالاً للأنمي (بالإنجليزية أو الرومانجي)
- CONFIDENCE: High أو Medium أو Low
- ALTERNATIVES: أنميات بديلة محتملة مفصولة بفواصل (حتى 5)
- إذا لم تتمكن من التعرف عليه، أعد 'UNKNOWN'

أمثلة:
- "شخص يقاتل عمالقة وجدران ضخمة" -> Attack on Titan|High|Kabaneri of the Iron Fortress, God Eater
- "ولد شعره أشقر يريد أن يصبح هوكاجي" -> Naruto|High|Boruto
- "قراصنة يبحثون عن كنز" -> One Piece|High|Black Lagoon, Pirates of the Caribbean
- "شخص يستخدم دفتر لقتل الناس" -> Death Note|High|Future Diary"""

    result = call_gemini_text(prompt)

    if result and result.upper() != 'UNKNOWN' and len(result) > 2:
        return result.strip()
    return None


def allowed_audio(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def validate_time_format(time_str):
    if not time_str:
        return None

    time_str = time_str.strip()
    parts = time_str.split(':')

    try:
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            if minutes < 0 or seconds < 0 or seconds >= 60:
                return None
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(
                parts[2])
            if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
                return None
            return hours * 3600 + minutes * 60 + seconds
        else:
            return None
    except ValueError:
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/logo.png')
def serve_logo():
    return send_file('logo.png', mimetype='image/png')


@app.route('/video-info', methods=['POST'])
def video_info():
    try:
        data = request.get_json()
        url = data.get('url', '')

        if not url:
            return jsonify({'error': 'الرجاء إدخال رابط الفيديو'}), 400

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'noplaylist': True,
            'force_generic_extractor': False,
            'cookiefile':
            'cookies.txt' if os.path.exists('cookies.txt') else None,
            'socket_timeout': 30,
            'retries': 3,
            'http_headers': {
                'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept':
                'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            },
        }

        if ydl_opts.get('cookiefile') is None:
            del ydl_opts['cookiefile']

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if not info:
                return jsonify({'error':
                                'لم يتم العثور على معلومات الفيديو'}), 400

            duration_seconds = info.get('duration') or 0
            if duration_seconds:
                hours = duration_seconds // 3600
                minutes = (duration_seconds % 3600) // 60
                seconds = duration_seconds % 60
                if hours > 0:
                    duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    duration_formatted = f"{minutes:02d}:{seconds:02d}"
            else:
                duration_formatted = "00:00"

            title = info.get('title') or info.get('fulltitle') or 'غير معروف'
            channel = info.get('uploader') or info.get('channel') or info.get(
                'uploader_id') or 'غير معروف'
            thumbnail = info.get('thumbnail') or ''
            if not thumbnail and info.get('thumbnails'):
                thumbnails = info.get('thumbnails', [])
                if thumbnails:
                    thumbnail = thumbnails[-1].get('url', '')

            return jsonify({
                'title': title,
                'duration': duration_formatted,
                'duration_seconds': duration_seconds,
                'thumbnail': thumbnail,
                'channel': channel
            })

    except Exception as e:
        logging.error(f"Video info error: {str(e)}")
        return jsonify({'error': f'خطأ في جلب معلومات الفيديو: {str(e)}'}), 400


@app.route('/process-video', methods=['POST'])
def process_video():
    temp_video = None
    output_file = None

    try:
        data = request.get_json()
        url = data.get('url', '')
        start_time = data.get('start_time', '00:00')
        end_time = data.get('end_time', '')

        if not url:
            return jsonify({'error': 'الرجاء إدخال رابط الفيديو'}), 400

        if not end_time:
            return jsonify({'error': 'الرجاء تحديد وقت النهاية'}), 400

        start_seconds = validate_time_format(start_time)
        end_seconds = validate_time_format(end_time)

        if start_seconds is None:
            return jsonify({
                'error':
                'صيغة وقت البداية غير صحيحة. استخدم الصيغة MM:SS أو HH:MM:SS'
            }), 400

        if end_seconds is None:
            return jsonify({
                'error':
                'صيغة وقت النهاية غير صحيحة. استخدم الصيغة MM:SS أو HH:MM:SS'
            }), 400

        if end_seconds <= start_seconds:
            return jsonify(
                {'error': 'وقت النهاية يجب أن يكون أكبر من وقت البداية'}), 400

        unique_id = str(uuid.uuid4())[:8]
        temp_video = os.path.join(UPLOAD_FOLDER, f'temp_video_{unique_id}.mp4')
        output_file = os.path.join(UPLOAD_FOLDER, f'clip_{unique_id}.mp4')

        ydl_opts = {
            'format':
            'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best',
            'merge_output_format': 'mp4',
            'outtmpl': temp_video.replace('.mp4', '.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 7200,
            'retries': 5,
            'fragment_retries': 5,
            'http_headers': {
                'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept':
                'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            },
        }

        if os.path.exists('cookies.txt'):
            ydl_opts['cookiefile'] = 'cookies.txt'

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        actual_temp_video = None
        for ext in ['mp4', 'mkv', 'webm', 'avi']:
            possible_file = temp_video.replace('.mp4', f'.{ext}')
            if os.path.exists(possible_file):
                actual_temp_video = possible_file
                break

        if not actual_temp_video:
            return jsonify({'error': 'فشل في تحميل الفيديو'}), 500

        temp_video = actual_temp_video

        duration = end_seconds - start_seconds

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-ss',
            str(start_seconds), '-i', temp_video, '-t',
            str(duration), '-c:v', 'libx264', '-c:a', 'aac', '-strict',
            'experimental', output_file
        ]

        subprocess.run(ffmpeg_cmd,
                       check=True,
                       capture_output=True,
                       timeout=7200)

        if os.path.exists(temp_video):
            os.remove(temp_video)
            temp_video = None

        @after_this_request
        def cleanup(response):
            try:
                if output_file and os.path.exists(output_file):
                    os.remove(output_file)
            except Exception:
                pass
            return response

        return send_file(
            output_file,
            as_attachment=True,
            download_name=
            f'clip_{start_time.replace(":", "-")}_{end_time.replace(":", "-")}.mp4',
            mimetype='video/mp4')

    except subprocess.CalledProcessError as e:
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)
        if output_file and os.path.exists(output_file):
            os.remove(output_file)
        return jsonify({
            'error':
            f'خطأ في معالجة الفيديو: {e.stderr.decode() if e.stderr else str(e)}'
        }), 500
    except Exception as e:
        if temp_video and os.path.exists(temp_video):
            os.remove(temp_video)
        if output_file and os.path.exists(output_file):
            os.remove(output_file)
        return jsonify({'error': f'خطأ: {str(e)}'}), 500


@app.route('/search-anime', methods=['POST'])
def search_anime():
    temp_path = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'الرجاء تحميل صورة'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400

        if not allowed_image(file.filename):
            return jsonify({
                'error':
                'نوع الملف غير مدعوم. الأنواع المدعومة: PNG, JPG, JPEG, GIF, WEBP'
            }), 400

        unique_id = str(uuid.uuid4())[:8]
        ext = file.filename.rsplit('.', 1)[1].lower()
        temp_path = os.path.join(UPLOAD_FOLDER,
                                 f'anime_search_{unique_id}.{ext}')
        file.save(temp_path)

        from urllib.parse import quote

        gemini_result = identify_anime_with_gemini(temp_path)
        if gemini_result:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

            anime_name = gemini_result
            episode_info = 'غير معروف'
            description = ''

            if '|' in gemini_result:
                parts = gemini_result.split('|')
                anime_name = parts[0].strip()
                if len(parts) > 1:
                    episode_info = parts[1].strip() if parts[1].strip().lower(
                    ) != 'unknown' else 'غير معروف'
                if len(parts) > 2:
                    description = parts[2].strip()

            search_links = {
                'anilist':
                f"https://anilist.co/search/anime?search={quote(anime_name)}",
                'myanimelist':
                f"https://myanimelist.net/anime.php?q={quote(anime_name)}",
                'crunchyroll':
                f"https://www.crunchyroll.com/search?q={quote(anime_name)}",
                'youtube':
                f"https://www.youtube.com/results?search_query={quote(anime_name + ' anime')}"
            }

            return jsonify({
                'found': True,
                'anime_name': anime_name,
                'episode': episode_info,
                'similarity': 'AI',
                'timestamp': '',
                'video_preview': '',
                'image_preview': '',
                'detection_method': 'gemini_ai',
                'description': description,
                'search_links': search_links,
                'search_link': search_links['anilist']
            })

        with open(temp_path, 'rb') as f:
            response = requests.post('https://api.trace.moe/search',
                                     files={'image': f},
                                     timeout=30)

        if response.status_code != 200:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                'found': False,
                'message':
                'فشل الاتصال بخدمة البحث. تأكد من إعداد مفاتيح Gemini API أو جرب لاحقاً.',
                'suggest_search_by_name': True
            })

        data = response.json()

        if data.get('error'):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({'error': data['error']}), 400

        results = data.get('result', [])

        if not results:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                'found': False,
                'message':
                'لم يتم العثور على نتائج. جرب استخدام صورة أوضح أو البحث بالاسم.',
                'suggest_search_by_name': True
            })

        top_result = results[0]
        similarity = top_result.get('similarity', 0)

        if similarity < ANIME_SIMILARITY_THRESHOLD:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                'found': False,
                'message':
                'لم يتم العثور على تطابق دقيق. الرجاء استخدام صورة أوضح أو جرب البحث بالاسم.',
                'suggest_search_by_name': True
            })

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        anilist_info = top_result.get('anilist', {})
        anime_name = 'غير معروف'

        if isinstance(anilist_info, dict):
            title_info = anilist_info.get('title', {})
            anime_name = title_info.get('english') or title_info.get(
                'romaji') or title_info.get('native') or 'غير معروف'
        elif isinstance(anilist_info, int):
            try:
                anilist_query = '''
                query ($id: Int) {
                    Media (id: $id, type: ANIME) {
                        title {
                            romaji
                            english
                            native
                        }
                    }
                }
                '''
                anilist_response = requests.post('https://graphql.anilist.co',
                                                 json={
                                                     'query': anilist_query,
                                                     'variables': {
                                                         'id': anilist_info
                                                     }
                                                 },
                                                 timeout=10)
                if anilist_response.status_code == 200:
                    anilist_data = anilist_response.json()
                    media = anilist_data.get('data', {}).get('Media', {})
                    title_info = media.get('title', {})
                    anime_name = title_info.get('english') or title_info.get(
                        'romaji') or title_info.get('native') or 'غير معروف'
            except Exception:
                anime_name = f'Anilist ID: {anilist_info}'

        episode = top_result.get('episode', 'غير معروف')
        from_time = top_result.get('from', 0)
        to_time = top_result.get('to', 0)

        def format_time(seconds):
            mins = int(seconds) // 60
            secs = int(seconds) % 60
            return f"{mins:02d}:{secs:02d}"

        result = {
            'found': True,
            'anime_name': anime_name,
            'episode': episode if episode else 'غير معروف',
            'similarity': round(similarity * 100, 2),
            'timestamp': f"{format_time(from_time)} - {format_time(to_time)}",
            'video_preview': top_result.get('video', ''),
            'image_preview': top_result.get('image', '')
        }

        return jsonify(result)

    except requests.Timeout:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify(
            {'error': 'انتهت مهلة الاتصال. الرجاء المحاولة مرة أخرى.'}), 500
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'خطأ: {str(e)}'}), 500


@app.route('/search-anime-by-name', methods=['POST'])
def search_anime_by_name():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        is_description = data.get('is_description', True)

        if not name:
            return jsonify({'error': 'الرجاء إدخال وصف الأنمي أو اسمه'}), 400

        search_term = name
        gemini_suggestion = None

        gemini_result = identify_anime_by_description(name)
        if gemini_result and gemini_result.upper() != 'UNKNOWN':
            if '|' in gemini_result:
                parts = gemini_result.split('|')
                search_term = parts[0].strip()
                confidence = parts[1].strip() if len(parts) > 1 else 'Medium'
                alternatives = parts[2].strip() if len(parts) > 2 else ''
                gemini_suggestion = {
                    'name':
                    search_term,
                    'confidence':
                    confidence,
                    'alternatives':
                    [a.strip() for a in alternatives.split(',') if a.strip()],
                    'original_input':
                    name
                }
            else:
                search_term = gemini_result.strip()
                gemini_suggestion = {
                    'name': search_term,
                    'confidence': 'Medium',
                    'alternatives': [],
                    'original_input': name
                }

        anilist_query = '''
        query ($search: String) {
            Page(page: 1, perPage: 5) {
                media(search: $search, type: ANIME, sort: POPULARITY_DESC) {
                    id
                    title {
                        romaji
                        english
                        native
                    }
                    description(asHtml: false)
                    coverImage {
                        large
                        medium
                    }
                    episodes
                    status
                    genres
                    averageScore
                    seasonYear
                    siteUrl
                }
            }
        }
        '''

        response = requests.post('https://graphql.anilist.co',
                                 json={
                                     'query': anilist_query,
                                     'variables': {
                                         'search': search_term
                                     }
                                 },
                                 timeout=15)

        if response.status_code != 200:
            return jsonify({'error': 'فشل الاتصال بخدمة البحث'}), 500

        data = response.json()
        media_list = data.get('data', {}).get('Page', {}).get('media', [])

        if not media_list:
            return jsonify({
                'found': False,
                'message':
                'لم يتم العثور على نتائج. جرب اسماً أو وصفاً مختلفاً.',
                'gemini_suggestion': gemini_suggestion
            })

        from urllib.parse import quote
        results = []
        for media in media_list:
            title_info = media.get('title', {})
            anime_name = title_info.get('english') or title_info.get(
                'romaji') or title_info.get('native') or 'غير معروف'

            description = media.get('description', '')
            if description:
                description = re.sub(r'<[^>]+>', '', description)
                if len(description) > 300:
                    description = description[:300] + '...'

            results.append({
                'id':
                media.get('id'),
                'name':
                anime_name,
                'name_native':
                title_info.get('native', ''),
                'description':
                description or 'لا يوجد وصف متاح',
                'cover':
                media.get('coverImage', {}).get('large')
                or media.get('coverImage', {}).get('medium', ''),
                'episodes':
                media.get('episodes') or 'غير محدد',
                'status':
                media.get('status', 'غير معروف'),
                'genres':
                media.get('genres', []),
                'score':
                media.get('averageScore') or 0,
                'year':
                media.get('seasonYear') or 'غير معروف',
                'anilist_url':
                media.get('siteUrl', ''),
                'search_links': {
                    'anilist':
                    media.get(
                        'siteUrl',
                        f"https://anilist.co/search/anime?search={quote(anime_name)}"
                    ),
                    'myanimelist':
                    f"https://myanimelist.net/anime.php?q={quote(anime_name)}",
                    'crunchyroll':
                    f"https://www.crunchyroll.com/search?q={quote(anime_name)}"
                }
            })

        return jsonify({
            'found': True,
            'results': results,
            'gemini_suggestion': gemini_suggestion
        })

    except requests.Timeout:
        return jsonify(
            {'error': 'انتهت مهلة الاتصال. الرجاء المحاولة مرة أخرى.'}), 500
    except Exception as e:
        return jsonify({'error': f'خطأ: {str(e)}'}), 500


@app.route('/transcribe-file', methods=['POST'])
def transcribe_file():
    temp_audio = None
    temp_wav = None
    chunk_files = []

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'الرجاء تحميل ملف صوتي'}), 400

        file = request.files['audio']
        language = request.form.get('language', 'ar')
        if language not in LANGUAGE_NAMES:
            language = 'ar'

        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400

        if not allowed_audio(file.filename):
            return jsonify({
                'error':
                'نوع الملف غير مدعوم. الأنواع المدعومة: MP3, WAV, OGG, M4A, FLAC, AAC, WMA'
            }), 400

        unique_id = str(uuid.uuid4())[:8]
        ext = file.filename.rsplit('.', 1)[1].lower()
        temp_audio = os.path.join(UPLOAD_FOLDER, f'audio_{unique_id}.{ext}')
        temp_wav = os.path.join(UPLOAD_FOLDER, f'audio_{unique_id}.wav')

        file.save(temp_audio)

        try:
            audio = AudioSegment.from_file(temp_audio)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            duration_seconds = len(audio) / 1000
            duration_minutes = duration_seconds / 60
            logging.info(
                f"Audio file loaded: {duration_minutes:.1f} minutes ({duration_seconds:.0f} seconds)"
            )
        except Exception as e:
            logging.error(f"Audio loading error: {str(e)}")
            return jsonify({'error':
                            f'خطأ في قراءة الملف الصوتي: {str(e)}'}), 400

        if duration_seconds > 7200:
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
            return jsonify({
                'error':
                'الملف الصوتي أطول من ساعتين. الحد الأقصى المدعوم هو ساعتين.'
            }), 400

        text = None
        transcription_method = 'gemini'

        if GEMINI_KEYS:
            logging.info(
                f"Audio duration: {duration_minutes:.1f} min - using Gemini AI with {get_available_keys_count()} available keys"
            )

            try:
                text = transcribe_audio_with_gemini(temp_audio, language)
            except ValueError as ve:
                if temp_audio and os.path.exists(temp_audio):
                    os.remove(temp_audio)
                return jsonify({'error': str(ve)}), 400
            except Exception as e:
                logging.error(f"Gemini transcription failed: {str(e)}")
                text = None

            if text:
                if temp_audio and os.path.exists(temp_audio):
                    os.remove(temp_audio)

                return jsonify({
                    'success': True,
                    'text': text,
                    'duration': round(duration_seconds, 1),
                    'duration_formatted': f"{int(duration_minutes)} دقيقة",
                    'method': 'gemini_ai',
                    'keys_available': get_available_keys_count()
                })

        if not text and not GEMINI_KEYS:
            transcription_method = 'google_speech'
            logging.info(
                f"No Gemini keys available, falling back to Google Speech Recognition for {duration_seconds}s audio"
            )

            if duration_seconds > 180:
                if temp_audio and os.path.exists(temp_audio):
                    os.remove(temp_audio)
                return jsonify({
                    'error':
                    'لتحويل ملفات أطول من 3 دقائق، يرجى إضافة مفاتيح Gemini API. الملفات الطويلة تحتاج إلى مفاتيح API.'
                }), 400

            try:
                audio.export(temp_wav, format="wav")
            except Exception as e:
                logging.error(f"Audio conversion error: {str(e)}")
                return jsonify(
                    {'error': f'خطأ في تحويل الملف الصوتي: {str(e)}'}), 400

            recognizer = sr.Recognizer()
            audio_segment = AudioSegment.from_wav(temp_wav)

            if duration_seconds > 180:
                audio_segment = audio_segment[:180 * 1000]
                duration_seconds = 180
                logging.info("Audio trimmed to 180 seconds for Google Speech")

            CHUNK_DURATION_MS = 30 * 1000

            if duration_seconds <= 30:
                with sr.AudioFile(temp_wav) as source:
                    audio_data = recognizer.record(source)

                try:
                    text = recognizer.recognize_google(audio_data,
                                                       language='ar-EG')
                except sr.UnknownValueError:
                    try:
                        text = recognizer.recognize_google(audio_data,
                                                           language='ar-SA')
                    except sr.UnknownValueError:
                        text = ''
                except sr.RequestError as e:
                    return jsonify(
                        {'error':
                         f'خطأ في خدمة التعرف على الصوت: {str(e)}'}), 500
            else:
                logging.info(f"Splitting audio into chunks for Google Speech")

                transcribed_texts = []
                num_chunks = int(len(audio_segment) / CHUNK_DURATION_MS) + 1

                for i in range(num_chunks):
                    start_ms = i * CHUNK_DURATION_MS
                    end_ms = min((i + 1) * CHUNK_DURATION_MS,
                                 len(audio_segment))
                    chunk = audio_segment[start_ms:end_ms]

                    chunk_path = os.path.join(UPLOAD_FOLDER,
                                              f'chunk_{unique_id}_{i}.wav')
                    chunk_files.append(chunk_path)
                    chunk.export(chunk_path, format="wav")

                    try:
                        with sr.AudioFile(chunk_path) as source:
                            chunk_audio = recognizer.record(source)

                        chunk_text = recognizer.recognize_google(
                            chunk_audio, language='ar-EG')
                        transcribed_texts.append(chunk_text)
                        logging.info(
                            f"Chunk {i+1}/{num_chunks} transcribed successfully"
                        )
                    except sr.UnknownValueError:
                        try:
                            chunk_text = recognizer.recognize_google(
                                chunk_audio, language='ar-SA')
                            transcribed_texts.append(chunk_text)
                            logging.info(
                                f"Chunk {i+1}/{num_chunks} transcribed with fallback"
                            )
                        except sr.UnknownValueError:
                            logging.info(
                                f"Chunk {i+1}/{num_chunks}: No speech detected"
                            )
                            continue
                    except sr.RequestError as e:
                        logging.error(
                            f"Chunk {i+1}/{num_chunks} error: {str(e)}")
                        continue

                text = ' '.join(transcribed_texts)

        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

        if not text:
            return jsonify({
                'success':
                True,
                'text':
                '',
                'message':
                'لم يتم التعرف على أي كلام في الملف الصوتي. تأكد من إعداد مفاتيح Gemini API للملفات الطويلة.'
            })

        return jsonify({
            'success': True,
            'text': text,
            'duration': round(duration_seconds, 1),
            'method': transcription_method
        })

    except ValueError as ve:
        logging.error(f"Transcription ValueError: {str(ve)}")
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        error_message = str(e)
        logging.error(f"Transcription error: {error_message}")
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

        if 'timeout' in error_message.lower(
        ) or 'timed out' in error_message.lower():
            return jsonify({
                'error':
                'انتهت مهلة المعالجة. جاري المحاولة مع مفتاح آخر أو حاول مرة أخرى.'
            }), 500
        elif 'quota' in error_message.lower() or 'rate' in error_message.lower(
        ):
            return jsonify({
                'error':
                'تم استنفاد حصة API. حاول مرة أخرى بعد ساعة أو أضف مفاتيح جديدة.'
            }), 500
        elif 'memory' in error_message.lower():
            return jsonify({'error':
                            'الملف كبير جداً. جرب ملف أصغر حجماً.'}), 500
        return jsonify({'error':
                        f'خطأ في التحويل: {error_message[:100]}'}), 500


@app.route('/ocr-image', methods=['POST'])
def ocr_image():
    temp_image = None

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'الرجاء تحميل صورة'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400

        if not allowed_image(file.filename):
            return jsonify({
                'error':
                'نوع الملف غير مدعوم. الأنواع المدعومة: PNG, JPG, JPEG, GIF, WEBP'
            }), 400

        unique_id = str(uuid.uuid4())[:8]
        ext = file.filename.rsplit('.', 1)[1].lower()
        temp_image = os.path.join(UPLOAD_FOLDER, f'ocr_{unique_id}.{ext}')

        file.save(temp_image)

        image = Image.open(temp_image)

        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image,
                                           lang='ara+eng',
                                           config=custom_config)

        if temp_image and os.path.exists(temp_image):
            os.remove(temp_image)

        if not text.strip():
            return jsonify({
                'success': True,
                'text': '',
                'message': 'لم يتم العثور على نص في الصورة'
            })

        return jsonify({'success': True, 'text': text.strip()})

    except Exception as e:
        if temp_image and os.path.exists(temp_image):
            os.remove(temp_image)
        return jsonify({'error': f'خطأ: {str(e)}'}), 500


@app.route('/search-podcast-by-image', methods=['POST'])
def search_podcast_by_image():
    temp_image = None

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'الرجاء تحميل صورة'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400

        if not allowed_image(file.filename):
            return jsonify({
                'error':
                'نوع الملف غير مدعوم. الأنواع المدعومة: PNG, JPG, JPEG, GIF, WEBP'
            }), 400

        unique_id = str(uuid.uuid4())[:8]
        ext = file.filename.rsplit('.', 1)[1].lower()
        temp_image = os.path.join(UPLOAD_FOLDER,
                                  f'podcast_ocr_{unique_id}.{ext}')

        file.save(temp_image)

        search_term = None
        extracted_text = ''
        detection_method = 'ai_vision'
        host_names = ''
        platform_hint = ''

        gemini_result = identify_podcast_with_gemini(temp_image)
        if gemini_result:
            if '|' in gemini_result:
                parts = gemini_result.split('|')
                search_term = parts[0].strip()
                extracted_text = search_term
                if len(parts) > 1:
                    host_names = parts[1].strip() if parts[1].strip().lower(
                    ) != 'unknown' else ''
                if len(parts) > 2:
                    platform_hint = parts[2].strip()
            else:
                search_term = gemini_result.strip()
                extracted_text = search_term
            detection_method = 'gemini_ai'

        if not search_term:
            image = Image.open(temp_image)
            extracted_text = pytesseract.image_to_string(
                image, lang='ara+eng', config='--oem 3 --psm 6')
            extracted_text = extracted_text.strip()
            detection_method = 'ocr'

            if extracted_text and len(extracted_text) > 3:
                search_term = ' '.join(extracted_text.split()[:10])

        if not search_term:
            if temp_image and os.path.exists(temp_image):
                os.remove(temp_image)
            return jsonify({
                'found':
                False,
                'message':
                'لم يتم التعرف على البودكاست. جرب صورة أوضح تحتوي على نص أو وجوه معروفة.'
            })

        itunes_response = requests.get('https://itunes.apple.com/search',
                                       params={
                                           'term': search_term,
                                           'media': 'podcast',
                                           'limit': 5
                                       },
                                       timeout=15)

        if itunes_response.status_code != 200:
            if temp_image and os.path.exists(temp_image):
                os.remove(temp_image)
            return jsonify({'error': 'فشل الاتصال بخدمة البحث'}), 500

        itunes_data = itunes_response.json()
        podcasts = itunes_data.get('results', [])

        smart_links = generate_podcast_search_links(search_term)

        if not podcasts:
            if temp_image and os.path.exists(temp_image):
                os.remove(temp_image)
            return jsonify({
                'found': False,
                'extracted_text': extracted_text,
                'detection_method': detection_method,
                'message':
                'لم يتم العثور على بودكاست مطابق في iTunes. جرب الروابط أدناه للبحث يدوياً.',
                'search_links': smart_links
            })

        best_match = None
        best_similarity = 0

        for podcast in podcasts:
            podcast_name = podcast.get('collectionName', '')
            artist_name = podcast.get('artistName', '')

            name_similarity = fuzz.ratio(extracted_text.lower(),
                                         podcast_name.lower())
            artist_similarity = fuzz.ratio(extracted_text.lower(),
                                           artist_name.lower())

            partial_name = fuzz.partial_ratio(extracted_text.lower(),
                                              podcast_name.lower())

            max_similarity = max(name_similarity, artist_similarity,
                                 partial_name)

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = podcast

        if best_similarity < PODCAST_NAME_SIMILARITY_THRESHOLD and detection_method == 'ocr':
            gemini_result = identify_podcast_with_gemini(temp_image)

            if gemini_result:
                detection_method = 'ai_vision'
                extracted_text = gemini_result

                itunes_response2 = requests.get(
                    'https://itunes.apple.com/search',
                    params={
                        'term': gemini_result,
                        'media': 'podcast',
                        'limit': 5
                    },
                    timeout=15)

                if itunes_response2.status_code == 200:
                    itunes_data2 = itunes_response2.json()
                    podcasts2 = itunes_data2.get('results', [])

                    if podcasts2:
                        best_match = None
                        best_similarity = 0

                        for podcast in podcasts2:
                            podcast_name = podcast.get('collectionName', '')
                            artist_name = podcast.get('artistName', '')

                            name_similarity = fuzz.ratio(
                                gemini_result.lower(), podcast_name.lower())
                            artist_similarity = fuzz.ratio(
                                gemini_result.lower(), artist_name.lower())
                            partial_name = fuzz.partial_ratio(
                                gemini_result.lower(), podcast_name.lower())

                            max_similarity = max(name_similarity,
                                                 artist_similarity,
                                                 partial_name)

                            if max_similarity > best_similarity:
                                best_similarity = max_similarity
                                best_match = podcast

                        smart_links = generate_podcast_search_links(
                            gemini_result)

            if not best_match or best_similarity < PODCAST_NAME_SIMILARITY_THRESHOLD:
                if temp_image and os.path.exists(temp_image):
                    os.remove(temp_image)
                return jsonify({
                    'found': False,
                    'extracted_text': extracted_text,
                    'detection_method': detection_method,
                    'similarity': best_similarity,
                    'message':
                    'تعذر التعرف على البودكاست بدقة عالية. جرب الروابط أدناه للبحث يدوياً.',
                    'search_links': smart_links
                })

        if temp_image and os.path.exists(temp_image):
            os.remove(temp_image)

        podcast_name = best_match.get('collectionName', 'غير معروف')
        final_smart_links = generate_podcast_search_links(podcast_name)

        result = {
            'found': True,
            'extracted_text': extracted_text,
            'detection_method': detection_method,
            'similarity': best_similarity,
            'host_names': host_names,
            'platform_hint': platform_hint,
            'podcast': {
                'name':
                podcast_name,
                'artist':
                best_match.get('artistName', 'غير معروف'),
                'artwork':
                best_match.get('artworkUrl600')
                or best_match.get('artworkUrl100', ''),
                'genre':
                best_match.get('primaryGenreName', 'غير محدد'),
                'episode_count':
                best_match.get('trackCount', 0),
                'feed_url':
                best_match.get('feedUrl', ''),
                'itunes_url':
                best_match.get('collectionViewUrl', '')
            },
            'search_links': final_smart_links
        }

        return jsonify(result)

    except requests.Timeout:
        if temp_image and os.path.exists(temp_image):
            os.remove(temp_image)
        return jsonify(
            {'error': 'انتهت مهلة الاتصال. الرجاء المحاولة مرة أخرى.'}), 500
    except Exception as e:
        if temp_image and os.path.exists(temp_image):
            os.remove(temp_image)
        return jsonify({'error': f'خطأ: {str(e)}'}), 500


@app.route('/search-podcast-by-audio', methods=['POST'])
def search_podcast_by_audio():
    temp_audio = None
    temp_wav = None

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'الرجاء تحميل ملف صوتي'}), 400

        file = request.files['audio']

        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400

        if not allowed_audio(file.filename):
            return jsonify({
                'error':
                'نوع الملف غير مدعوم. الأنواع المدعومة: MP3, WAV, OGG, M4A, FLAC, AAC, WMA'
            }), 400

        unique_id = str(uuid.uuid4())[:8]
        ext = file.filename.rsplit('.', 1)[1].lower()
        temp_audio = os.path.join(UPLOAD_FOLDER,
                                  f'podcast_audio_{unique_id}.{ext}')
        temp_wav = os.path.join(UPLOAD_FOLDER,
                                f'podcast_audio_{unique_id}.wav')

        file.save(temp_audio)

        try:
            audio = AudioSegment.from_file(temp_audio)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            if len(audio) > 60000:
                audio = audio[:60000]

            audio.export(temp_wav, format="wav")
        except Exception as e:
            return jsonify({'error':
                            f'خطأ في تحويل الملف الصوتي: {str(e)}'}), 400

        recognizer = sr.Recognizer()

        with sr.AudioFile(temp_wav) as source:
            audio_data = recognizer.record(source)

        transcribed_text = ''
        try:
            transcribed_text = recognizer.recognize_google(audio_data,
                                                           language='ar-SA')
        except sr.UnknownValueError:
            try:
                transcribed_text = recognizer.recognize_google(
                    audio_data, language='en-US')
            except sr.UnknownValueError:
                transcribed_text = ''
        except sr.RequestError as e:
            return jsonify(
                {'error': f'خطأ في خدمة التعرف على الصوت: {str(e)}'}), 500

        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)

        if not transcribed_text:
            return jsonify({
                'found':
                False,
                'message':
                'لم يتم التعرف على أي كلام في الملف الصوتي. جرب ملفاً أوضح.'
            })

        gemini_podcast_name = identify_podcast_from_transcript(
            transcribed_text)

        search_term = gemini_podcast_name if gemini_podcast_name else ' '.join(
            transcribed_text.split()[:15])
        detection_method = 'gemini_ai' if gemini_podcast_name else 'keyword_match'

        itunes_response = requests.get('https://itunes.apple.com/search',
                                       params={
                                           'term': search_term,
                                           'media': 'podcast',
                                           'limit': 5
                                       },
                                       timeout=15)

        smart_links = generate_podcast_search_links(search_term)

        if itunes_response.status_code != 200:
            return jsonify({
                'found': False,
                'transcribed_text': transcribed_text,
                'identified_name': gemini_podcast_name,
                'detection_method': detection_method,
                'message':
                'فشل الاتصال بخدمة iTunes. استخدم الروابط أدناه للبحث يدوياً.',
                'search_links': smart_links
            })

        itunes_data = itunes_response.json()
        podcasts = itunes_data.get('results', [])

        if not podcasts:
            return jsonify({
                'found': False,
                'transcribed_text': transcribed_text,
                'identified_name': gemini_podcast_name,
                'detection_method': detection_method,
                'message':
                'لم يتم العثور على بودكاست مطابق. استخدم الروابط أدناه للبحث يدوياً.',
                'search_links': smart_links
            })

        results = []
        for podcast in podcasts:
            podcast_name = podcast.get('collectionName', 'غير معروف')
            podcast_links = generate_podcast_search_links(podcast_name)
            results.append({
                'name':
                podcast_name,
                'artist':
                podcast.get('artistName', 'غير معروف'),
                'artwork':
                podcast.get('artworkUrl600')
                or podcast.get('artworkUrl100', ''),
                'genre':
                podcast.get('primaryGenreName', 'غير محدد'),
                'episode_count':
                podcast.get('trackCount', 0),
                'feed_url':
                podcast.get('feedUrl', ''),
                'itunes_url':
                podcast.get('collectionViewUrl', ''),
                'search_links':
                podcast_links
            })

        return jsonify({
            'found': True,
            'transcribed_text': transcribed_text,
            'identified_name': gemini_podcast_name,
            'detection_method': detection_method,
            'results': results,
            'search_links': smart_links
        })

    except requests.Timeout:
        return jsonify(
            {'error': 'انتهت مهلة الاتصال. الرجاء المحاولة مرة أخرى.'}), 500
    except Exception as e:
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
        return jsonify({'error': f'خطأ: {str(e)}'}), 500


def cleanup_download_files(output_template, output_file):
    """Clean up all possible temp files from download"""
    try:
        for ext in [
                'mp4', 'mkv', 'webm', 'avi', 'mov', 'mp3', 'wav', 'ogg', 'm4a',
                'part', 'ytdl', 'temp'
        ]:
            temp_file = f'{output_template}.{ext}'
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if output_file and os.path.exists(output_file):
            os.remove(output_file)
    except Exception:
        pass


@app.route('/download-video', methods=['POST'])
def download_video():
    output_file = None
    output_template = None
    media_title = 'media'

    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        download_format = data.get('format', 'video')

        if not url:
            return jsonify({'error': 'الرجاء إدخال رابط الفيديو'}), 400

        if not url.startswith(('http://', 'https://')):
            return jsonify({
                'error':
                'الرابط غير صالح. يجب أن يبدأ بـ http:// أو https://'
            }), 400

        unique_id = str(uuid.uuid4())[:8]
        output_template = os.path.join(UPLOAD_FOLDER, f'download_{unique_id}')

        if download_format == 'audio':
            output_file = f'{output_template}.mp3'
        else:
            output_file = f'{output_template}.mp4'

        info_opts = {
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 60,
            'noplaylist': True,
            'retries': 3,
            'http_headers': {
                'User-Agent':
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept':
                'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            },
        }

        if os.path.exists('cookies.txt'):
            info_opts['cookiefile'] = 'cookies.txt'

        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if not info:
                return jsonify(
                    {'error': 'لم يتم العثور على محتوى في هذا الرابط'}), 400

            if info.get('_type') == 'playlist' or 'entries' in info:
                return jsonify({
                    'error':
                    'لا يمكن تحميل قوائم التشغيل. الرجاء استخدام رابط واحد.'
                }), 400

            media_title = info.get('title', 'media')
            media_title = re.sub(r'[\\/*?:"<>|]', '', media_title)[:50]

            if info.get('is_live'):
                return jsonify({'error': 'لا يمكن تحميل البث المباشر.'}), 400

        is_tiktok = 'tiktok.com' in url.lower() or 'douyin.com' in url.lower()
        is_instagram = 'instagram.com' in url.lower()
        is_twitter = 'twitter.com' in url.lower() or 'x.com' in url.lower()

        common_headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept':
            'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        }

        cookiefile_opt = 'cookies.txt' if os.path.exists(
            'cookies.txt') else None

        if download_format == 'audio':
            ydl_opts = {
                'format':
                'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best',
                'outtmpl':
                output_template + '.%(ext)s',
                'noplaylist':
                True,
                'quiet':
                True,
                'no_warnings':
                True,
                'socket_timeout':
                7200,
                'retries':
                5,
                'http_headers':
                common_headers,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '128',
                }],
            }
            if cookiefile_opt:
                ydl_opts['cookiefile'] = cookiefile_opt
        else:
            if is_tiktok or is_instagram or is_twitter:
                ydl_opts = {
                    'format':
                    'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio/bestvideo[ext=mp4]+bestaudio/best[ext=mp4][vcodec^=avc1]/best[ext=mp4]/best',
                    'merge_output_format':
                    'mp4',
                    'outtmpl':
                    output_template + '.%(ext)s',
                    'noplaylist':
                    True,
                    'quiet':
                    True,
                    'no_warnings':
                    True,
                    'socket_timeout':
                    7200,
                    'retries':
                    10,
                    'fragment_retries':
                    10,
                    'http_headers':
                    common_headers,
                    'extractor_args': {
                        'tiktok': {
                            'api_hostname':
                            'api22-normal-c-useast1a.tiktokv.com'
                        }
                    },
                    'postprocessors': [{
                        'key': 'FFmpegVideoConvertor',
                        'preferedformat': 'mp4',
                    }],
                    'postprocessor_args': {
                        'FFmpegVideoConvertor': [
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                            '-c:a', 'aac', '-b:a', '128k', '-movflags',
                            '+faststart'
                        ]
                    },
                }
                if cookiefile_opt:
                    ydl_opts['cookiefile'] = cookiefile_opt
            else:
                ydl_opts = {
                    'format':
                    'bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
                    'merge_output_format':
                    'mp4',
                    'outtmpl':
                    output_template + '.%(ext)s',
                    'noplaylist':
                    True,
                    'quiet':
                    True,
                    'no_warnings':
                    True,
                    'socket_timeout':
                    7200,
                    'retries':
                    5,
                    'fragment_retries':
                    5,
                    'http_headers':
                    common_headers,
                    'postprocessors': [{
                        'key': 'FFmpegVideoConvertor',
                        'preferedformat': 'mp4',
                    }],
                    'postprocessor_args': {
                        'FFmpegVideoConvertor': [
                            '-c:v', 'libx264', '-c:a', 'aac', '-movflags',
                            '+faststart'
                        ]
                    },
                }
                if cookiefile_opt:
                    ydl_opts['cookiefile'] = cookiefile_opt

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if download_format == 'audio':
            if not os.path.exists(output_file):
                for ext in ['mp3', 'wav', 'm4a', 'ogg', 'opus', 'webm']:
                    alt_file = f'{output_template}.{ext}'
                    if os.path.exists(alt_file):
                        if ext != 'mp3':
                            try:
                                ffmpeg_cmd = [
                                    'ffmpeg', '-y', '-i', alt_file, '-vn',
                                    '-acodec', 'libmp3lame', '-ab', '128k',
                                    '-threads', '4', '-q:a', '5', output_file
                                ]
                                subprocess.run(ffmpeg_cmd,
                                               check=True,
                                               capture_output=True,
                                               timeout=7200)
                                os.remove(alt_file)
                            except subprocess.CalledProcessError:
                                cleanup_download_files(output_template,
                                                       output_file)
                                return jsonify({
                                    'error':
                                    'فشل في تحويل الصوت إلى MP3. جرب رابطاً آخر.'
                                }), 500
                            except subprocess.TimeoutExpired:
                                cleanup_download_files(output_template,
                                                       output_file)
                                return jsonify({
                                    'error':
                                    'انتهت مهلة تحويل الصوت. جرب محتوى أقصر.'
                                }), 500
                        else:
                            output_file = alt_file
                        break
        else:
            if not os.path.exists(output_file):
                for ext in ['mp4', 'mkv', 'webm', 'avi', 'mov']:
                    alt_file = f'{output_template}.{ext}'
                    if os.path.exists(alt_file):
                        needs_conversion = False
                        if ext != 'mp4':
                            needs_conversion = True
                        else:
                            try:
                                probe_cmd = [
                                    'ffprobe', '-v', 'error',
                                    '-select_streams', 'v:0', '-show_entries',
                                    'stream=codec_name', '-of',
                                    'default=noprint_wrappers=1:nokey=1',
                                    alt_file
                                ]
                                result = subprocess.run(probe_cmd,
                                                        capture_output=True,
                                                        text=True,
                                                        timeout=30)
                                codec = result.stdout.strip().lower()
                                if codec in [
                                        'hevc', 'h265', 'hev1', 'hvc1', 'vp9',
                                        'av1'
                                ]:
                                    needs_conversion = True
                                    logging.info(
                                        f"Detected {codec} codec, converting to H.264 for compatibility"
                                    )
                            except Exception as probe_error:
                                logging.warning(
                                    f"Could not probe video codec: {probe_error}"
                                )
                                needs_conversion = True

                        if needs_conversion:
                            try:
                                ffmpeg_cmd = [
                                    'ffmpeg', '-y', '-i', alt_file, '-c:v',
                                    'libx264', '-preset', 'fast', '-crf', '23',
                                    '-c:a', 'aac', '-b:a', '128k', '-movflags',
                                    '+faststart', '-strict', 'experimental',
                                    output_file
                                ]
                                subprocess.run(ffmpeg_cmd,
                                               check=True,
                                               capture_output=True,
                                               timeout=7200)
                                if alt_file != output_file and os.path.exists(
                                        alt_file):
                                    os.remove(alt_file)
                            except subprocess.CalledProcessError:
                                cleanup_download_files(output_template,
                                                       output_file)
                                return jsonify({
                                    'error':
                                    'فشل في تحويل الفيديو إلى MP4. جرب رابطاً آخر.'
                                }), 500
                            except subprocess.TimeoutExpired:
                                cleanup_download_files(output_template,
                                                       output_file)
                                return jsonify({
                                    'error':
                                    'انتهت مهلة تحويل الفيديو. جرب فيديو أقصر.'
                                }), 500
                        else:
                            output_file = alt_file
                        break

        if not os.path.exists(output_file):
            cleanup_download_files(output_template, output_file)
            if download_format == 'audio':
                return jsonify(
                    {'error': 'فشل في تحميل الصوت. جرب رابطاً آخر.'}), 500
            else:
                return jsonify(
                    {'error': 'فشل في تحميل الفيديو. جرب رابطاً آخر.'}), 500

        final_output = output_file
        final_template = output_template

        @after_this_request
        def cleanup(response):
            cleanup_download_files(final_template, final_output)
            return response

        if download_format == 'audio':
            return send_file(output_file,
                             as_attachment=True,
                             download_name=f'{media_title}.mp3',
                             mimetype='audio/mpeg')
        else:
            return send_file(output_file,
                             as_attachment=True,
                             download_name=f'{media_title}.mp4',
                             mimetype='video/mp4')

    except yt_dlp.utils.DownloadError as e:
        cleanup_download_files(output_template, output_file)
        error_msg = str(e).lower()
        if 'private' in error_msg:
            return jsonify({'error': 'هذا الفيديو خاص ولا يمكن تحميله.'}), 400
        elif 'unavailable' in error_msg or 'removed' in error_msg or 'deleted' in error_msg:
            return jsonify({'error': 'الفيديو غير متاح أو تم حذفه.'}), 400
        elif 'unsupported' in error_msg or 'no video formats' in error_msg:
            return jsonify(
                {'error': 'هذا الموقع غير مدعوم أو الرابط غير صحيح.'}), 400
        elif '403' in error_msg or 'forbidden' in error_msg:
            return jsonify({'error':
                            'تم رفض الوصول. الفيديو قد يكون محمياً.'}), 400
        elif 'geo' in error_msg or 'country' in error_msg:
            return jsonify({'error': 'هذا الفيديو غير متاح في منطقتك.'}), 400
        elif 'drm' in error_msg or 'protected' in error_msg:
            return jsonify(
                {'error':
                 'هذا الفيديو محمي بحقوق النشر ولا يمكن تحميله.'}), 400
        elif 'age' in error_msg or 'sign in' in error_msg:
            return jsonify({'error': 'هذا الفيديو يتطلب تسجيل الدخول.'}), 400
        else:
            return jsonify({'error': f'خطأ في التحميل: {str(e)[:100]}'}), 400
    except Exception as e:
        cleanup_download_files(output_template, output_file)
        return jsonify({'error': f'خطأ غير متوقع: {str(e)[:100]}'}), 500


def extract_arabic_text_with_gemini(image):
    """Use Gemini AI to extract Arabic text from PDF page image with high accuracy"""
    prompt = """أنت أداة OCR متخصصة في استخراج النص العربي. قم باستخراج كل النص الموجود في هذه الصورة بدقة عالية.

التعليمات المهمة:
- استخرج النص العربي كما هو بالضبط، حرف بحرف وكلمة بكلمة
- حافظ على ترتيب الفقرات والسطور كما تظهر في الصورة
- لا تترجم أي نص - فقط استخرجه كما هو
- إذا كان هناك نص إنجليزي أو أرقام، استخرجها أيضاً كما هي
- افصل بين الفقرات بسطر فارغ
- لا تضف أي تعليقات أو شروحات - فقط النص المستخرج
- إذا لم يكن هناك نص مقروء، أرجع "لا يوجد نص"

أعد النص المستخرج فقط بدون أي مقدمات أو ملاحظات."""

    result = call_gemini_vision(image, prompt)

    if result and result.strip() and result.strip() != "لا يوجد نص":
        return result.strip()
    return None


def convert_pdf_with_ocr(pdf_path, docx_path, use_gemini=True):
    """Convert PDF to DOCX using OCR for better Arabic text extraction"""
    import fitz
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    try:
        pdf_doc = fitz.open(pdf_path)
        doc = Document()

        style = doc.styles['Normal']
        style.font.size = Pt(12)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.RIGHT

        total_pages = len(pdf_doc)
        extracted_any_text = False

        for page_num in range(total_pages):
            page = pdf_doc[page_num]

            if use_gemini and GEMINI_KEYS:
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                from io import BytesIO
                img = Image.open(BytesIO(img_data))

                extracted_text = extract_arabic_text_with_gemini(img)

                if extracted_text:
                    extracted_any_text = True
                    paragraphs = extracted_text.split('\n\n')
                    for para_text in paragraphs:
                        if para_text.strip():
                            p = doc.add_paragraph()
                            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                            run = p.add_run(para_text.strip())
                            run.font.size = Pt(12)
            else:
                text = page.get_text("text")
                if text and text.strip():
                    extracted_any_text = True
                    paragraphs = text.split('\n\n')
                    for para_text in paragraphs:
                        if para_text.strip():
                            p = doc.add_paragraph()
                            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                            run = p.add_run(para_text.strip())
                            run.font.size = Pt(12)

            if page_num < total_pages - 1:
                doc.add_page_break()

        pdf_doc.close()

        if not extracted_any_text:
            return False, "لم يتم استخراج أي نص من الملف"

        doc.save(docx_path)
        return True, None

    except Exception as e:
        logging.error(f"OCR conversion error: {e}")
        return False, str(e)


def has_arabic_text_issues(text):
    """Check if extracted text has common Arabic encoding issues"""
    if not text:
        return True

    issue_patterns = [
        '\ufffd',
        '�',
        '\x00',
    ]

    for pattern in issue_patterns:
        if pattern in text:
            return True

    arabic_char_count = sum(
        1 for c in text
        if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
    total_chars = len(text.replace(' ', '').replace('\n', ''))

    if total_chars > 0 and arabic_char_count / total_chars < 0.1:
        has_arabic_looking = any('\u0600' <= c <= '\u06FF' for c in text)
        if not has_arabic_looking:
            return True

    return False


@app.route('/convert-pdf', methods=['POST'])
def convert_pdf():
    pdf_path = None
    docx_path = None

    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'لم يتم تحميل ملف PDF'}), 400

        pdf_file = request.files['pdf']
        mode = request.form.get('mode', 'original')
        extraction_method = request.form.get('extraction_method', 'auto')

        if pdf_file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400

        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'الرجاء تحميل ملف PDF فقط'}), 400

        unique_id = str(uuid.uuid4())[:8]
        pdf_path = os.path.join(UPLOAD_FOLDER, f'input_{unique_id}.pdf')
        docx_path = os.path.join(UPLOAD_FOLDER, f'output_{unique_id}.docx')

        pdf_file.save(pdf_path)

        conversion_success = False
        use_ocr = extraction_method == 'ocr' or extraction_method == 'auto'

        if extraction_method == 'auto' or extraction_method == 'standard':
            try:
                from pdf2docx import Converter
                cv = Converter(pdf_path)
                cv.convert(docx_path, start=0, end=None)
                cv.close()

                if os.path.exists(docx_path):
                    from docx import Document
                    test_doc = Document(docx_path)
                    all_text = ""
                    for para in test_doc.paragraphs:
                        all_text += para.text + " "

                    if extraction_method == 'auto' and has_arabic_text_issues(
                            all_text):
                        logging.info(
                            "Standard conversion has Arabic issues, trying OCR..."
                        )
                        os.remove(docx_path)
                        use_ocr = True
                    else:
                        conversion_success = True
                        use_ocr = False

            except Exception as e:
                logging.warning(f"Standard PDF conversion failed: {e}")
                if extraction_method == 'standard':
                    if pdf_path and os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    return jsonify(
                        {'error': f'فشل في تحويل PDF: {str(e)[:100]}'}), 500
                use_ocr = True

        if use_ocr and not conversion_success:
            if GEMINI_KEYS:
                logging.info(
                    "Using Gemini AI OCR for Arabic text extraction...")
                success, error = convert_pdf_with_ocr(pdf_path,
                                                      docx_path,
                                                      use_gemini=True)
                if success:
                    conversion_success = True
                else:
                    logging.warning(f"Gemini OCR failed: {error}")

            if not conversion_success:
                success, error = convert_pdf_with_ocr(pdf_path,
                                                      docx_path,
                                                      use_gemini=False)
                if success:
                    conversion_success = True
                else:
                    if pdf_path and os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    return jsonify({
                        'error':
                        f'فشل في استخراج النص: {error[:100] if error else "خطأ غير معروف"}'
                    }), 500

        if mode == 'en':
            try:
                from docx import Document
                from deep_translator import GoogleTranslator

                doc = Document(docx_path)
                translator = GoogleTranslator(source='auto', target='en')

                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        try:
                            translated_text = translator.translate(
                                paragraph.text)
                            if translated_text:
                                paragraph.text = translated_text
                        except Exception:
                            pass

                for table in doc.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            for paragraph in cell.paragraphs:
                                if paragraph.text.strip():
                                    try:
                                        translated_text = translator.translate(
                                            paragraph.text)
                                        if translated_text:
                                            paragraph.text = translated_text
                                    except Exception:
                                        pass

                doc.save(docx_path)

            except Exception as e:
                if pdf_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)
                if docx_path and os.path.exists(docx_path):
                    os.remove(docx_path)
                return jsonify({'error':
                                f'فشل في الترجمة: {str(e)[:100]}'}), 500

        if not os.path.exists(docx_path):
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)
            return jsonify({'error': 'فشل في إنشاء ملف Word'}), 500

        final_pdf = pdf_path
        final_docx = docx_path

        @after_this_request
        def cleanup(response):
            try:
                if final_pdf and os.path.exists(final_pdf):
                    os.remove(final_pdf)
                if final_docx and os.path.exists(final_docx):
                    os.remove(final_docx)
            except Exception:
                pass
            return response

        return send_file(
            docx_path,
            as_attachment=True,
            download_name='converted_document.docx',
            mimetype=
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception:
                pass
        if docx_path and os.path.exists(docx_path):
            try:
                os.remove(docx_path)
            except Exception:
                pass
        return jsonify({'error': f'خطأ غير متوقع: {str(e)[:100]}'}), 500


ALLOWED_VIDEO_EXTENSIONS = {
    'mp4', 'mov', 'avi', 'mkv', 'webm', 'flv', 'wmv', 'm4v'
}


def allowed_video(filename):
    return '.' in filename and filename.rsplit(
        '.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video file using ffmpeg - optimized for speed"""
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame',
            '-q:a', '5', '-threads', '4', '-ar', '16000', '-ac', '1',
            output_audio_path
        ]
        subprocess.run(ffmpeg_cmd,
                       check=True,
                       capture_output=True,
                       timeout=7200)
        return output_audio_path
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg audio extraction error: {e}")
        return None
    except subprocess.TimeoutExpired:
        logging.error("FFmpeg audio extraction timed out")
        return None


@app.route('/transcribe-video', methods=['POST'])
def transcribe_video():
    temp_video = None
    temp_audio = None
    compressed_audio = None

    try:
        input_type = request.form.get('input_type', 'file')
        language = request.form.get('language', 'ar')
        if language not in LANGUAGE_NAMES:
            language = 'ar'

        logging.info(
            f"Video transcription started - input_type: {input_type}, language: {language}"
        )
        logging.info(
            f"Available Gemini keys: {get_available_keys_count()}/{len(GEMINI_KEYS)}"
        )

        if input_type == 'url':
            url = request.form.get('url', '').strip()

            if not url:
                return jsonify({'error': 'الرجاء إدخال رابط الفيديو'}), 400

            if not url.startswith(('http://', 'https://')):
                return jsonify({
                    'error':
                    'الرابط غير صالح. يجب أن يبدأ بـ http:// أو https://'
                }), 400

            unique_id = str(uuid.uuid4())[:8]
            temp_audio = os.path.join(UPLOAD_FOLDER,
                                      f'video_audio_{unique_id}.mp3')

            ydl_opts = {
                'format':
                'bestaudio/best',
                'outtmpl':
                temp_audio.replace('.mp3', '.%(ext)s'),
                'noplaylist':
                True,
                'quiet':
                True,
                'no_warnings':
                True,
                'socket_timeout':
                300,
                'retries':
                15,
                'fragment_retries':
                15,
                'file_access_retries':
                10,
                'extractor_retries':
                10,
                'buffersize':
                1024,
                'http_chunk_size':
                10485760,
                'http_headers': {
                    'User-Agent':
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept':
                    'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                },
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '64',
                }],
            }

            if os.path.exists('cookies.txt'):
                ydl_opts['cookiefile'] = 'cookies.txt'

            max_download_attempts = 5
            download_success = False
            last_error = None

            for attempt in range(max_download_attempts):
                try:
                    logging.info(
                        f"Download attempt {attempt + 1}/{max_download_attempts} for URL: {url[:50]}..."
                    )
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                    download_success = True
                    logging.info(
                        f"Download successful on attempt {attempt + 1}")
                    break
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    if 'private' in error_msg:
                        return jsonify(
                            {'error':
                             'هذا الفيديو خاص ولا يمكن الوصول إليه.'}), 400
                    elif 'unavailable' in error_msg or 'removed' in error_msg or 'deleted' in error_msg:
                        return jsonify(
                            {'error': 'الفيديو غير متاح أو تم حذفه.'}), 400
                    elif 'age' in error_msg or 'sign in' in error_msg:
                        return jsonify(
                            {'error': 'هذا الفيديو يتطلب تسجيل الدخول.'}), 400
                    else:
                        logging.warning(
                            f"Download attempt {attempt + 1} failed: {str(e)[:100]}"
                        )
                        if attempt < max_download_attempts - 1:
                            import time
                            wait_time = (attempt + 1) * 3
                            logging.info(
                                f"Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                            continue

            if not download_success:
                return jsonify({
                    'error':
                    f'فشل في تحميل الفيديو بعد {max_download_attempts} محاولات. جرب مرة أخرى أو استخدم رابط آخر.'
                }), 400

            if not os.path.exists(temp_audio):
                for ext in ['mp3', 'wav', 'm4a', 'ogg', 'opus', 'webm']:
                    alt_file = temp_audio.replace('.mp3', f'.{ext}')
                    if os.path.exists(alt_file):
                        if ext != 'mp3':
                            new_temp_audio = os.path.join(
                                UPLOAD_FOLDER,
                                f'video_audio_{unique_id}_converted.mp3')
                            try:
                                ffmpeg_cmd = [
                                    'ffmpeg', '-y', '-i', alt_file, '-acodec',
                                    'libmp3lame', '-ab', '128k', new_temp_audio
                                ]
                                subprocess.run(ffmpeg_cmd,
                                               check=True,
                                               capture_output=True,
                                               timeout=7200)
                                os.remove(alt_file)
                                temp_audio = new_temp_audio
                            except:
                                temp_audio = alt_file
                        else:
                            temp_audio = alt_file
                        break

            if not os.path.exists(temp_audio):
                return jsonify({'error':
                                'فشل في استخراج الصوت من الفيديو'}), 500

        else:
            if 'video' not in request.files:
                return jsonify({'error': 'الرجاء تحميل ملف فيديو'}), 400

            file = request.files['video']

            if file.filename == '':
                return jsonify({'error': 'لم يتم اختيار ملف'}), 400

            if not allowed_video(file.filename):
                return jsonify({
                    'error':
                    'نوع الملف غير مدعوم. الأنواع المدعومة: MP4, MOV, AVI, MKV, WEBM, FLV, WMV, M4V'
                }), 400

            unique_id = str(uuid.uuid4())[:8]
            ext = file.filename.rsplit('.', 1)[1].lower()
            temp_video = os.path.join(UPLOAD_FOLDER,
                                      f'video_{unique_id}.{ext}')
            temp_audio = os.path.join(UPLOAD_FOLDER,
                                      f'video_audio_{unique_id}.mp3')

            file.save(temp_video)

            extracted_audio = extract_audio_from_video(temp_video, temp_audio)

            if not extracted_audio or not os.path.exists(temp_audio):
                if temp_video and os.path.exists(temp_video):
                    os.remove(temp_video)
                return jsonify({
                    'error':
                    'فشل في استخراج الصوت من الفيديو. تأكد من أن الفيديو يحتوي على صوت.'
                }), 400

            if temp_video and os.path.exists(temp_video):
                os.remove(temp_video)
                temp_video = None

        logging.info(
            f"Starting Gemini transcription for video audio: {temp_audio} with language: {language}"
        )

        try:
            text = transcribe_audio_with_gemini(temp_audio, language)
        except ValueError as ve:
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
            return jsonify({'error': str(ve)}), 400
        except Exception as transcribe_error:
            logging.error(f"Transcription error: {transcribe_error}")
            if temp_audio and os.path.exists(temp_audio):
                os.remove(temp_audio)
            return jsonify({
                'error':
                'فشل في تحويل الصوت إلى نص. حاول مرة أخرى أو استخدم ملف أقصر.'
            }), 500

        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)

        if text:
            word_count = len(text.split())
            return jsonify({
                'success': True,
                'text': text,
                'method': 'gemini_ai',
                'word_count': word_count,
                'keys_available': get_available_keys_count()
            })
        else:
            return jsonify({
                'error':
                'فشل في تحويل الفيديو إلى نص. تأكد من إعداد مفاتيح Gemini API أو جرب ملف أقصر.'
            }), 500

    except ValueError as ve:
        if temp_video and os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except:
                pass
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except:
                pass
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logging.error(f"Video transcription error: {e}")
        if temp_video and os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except:
                pass
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except:
                pass
        error_message = str(e)
        if 'timeout' in error_message.lower(
        ) or 'timed out' in error_message.lower():
            return jsonify({
                'error':
                'انتهت مهلة المعالجة. جرب ملف أقصر أو اتصال إنترنت أسرع.'
            }), 500
        elif 'memory' in error_message.lower():
            return jsonify({'error':
                            'الملف كبير جداً. جرب ملف أصغر حجماً.'}), 500
        elif 'quota' in error_message.lower() or 'rate' in error_message.lower(
        ):
            return jsonify(
                {'error': 'تم استنفاد حصة API. حاول مرة أخرى لاحقاً.'}), 500
        return jsonify(
            {'error': f'حدث خطأ أثناء المعالجة: {error_message[:80]}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
