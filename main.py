import os
import re
import uuid
import tempfile
import subprocess
import base64
import logging
import gc
import time
import glob
import threading
from datetime import datetime, date, timedelta
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_file, after_this_request, redirect, url_for, flash
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import yt_dlp
import requests
import speech_recognition as sr
from pydub import AudioSegment
import pytesseract
from PIL import Image
from fuzzywuzzy import fuzz
import google.generativeai as genai
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from models import db, AdminUser, GeminiKeyState, GeminiUsageLog, DailyStats

# --- كود إنشاء ملف الكوكيز تلقائياً من إعدادات السيرفر ---
# هذا يحمي حسابك بدلاً من رفع الملف على GitHub
cookie_content = os.environ.get('COOKIE_CONTENT')
if cookie_content:
    with open('cookies.txt', 'w') as f:
        f.write(cookie_content)
    print("✅ Cookies file created successfully from environment variables.")
else:
    print("⚠️ Warning: COOKIE_CONTENT not found in environment variables.")
# -------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# MEMORY OPTIMIZATION: Centralized cleanup utilities
# =============================================================================
TEMP_FILE_PATTERNS = [
    'audio_*.mp3', 'audio_*.wav', 'audio_*.m4a', 'audio_*.webm',
    'video_*.mp4', 'video_*.webm', 'video_*.mkv',
    'ocr_*.*', 'cut_*.*', 'arkan_*.*',
    '*_compressed.mp3', '*_chunk_*.wav'
]
TEMP_FILE_MAX_AGE_SECONDS = 3600
last_cleanup_time = 0

def cleanup_temp_files(force=False):
    """
    Clean up old temporary files to prevent disk space issues.
    
    Args:
        force: If True, clean all matching files regardless of age.
               If False, only clean files older than TEMP_FILE_MAX_AGE_SECONDS.
    """
    global last_cleanup_time
    current_time = time.time()
    
    if not force and current_time - last_cleanup_time < 300:
        return
    
    last_cleanup_time = current_time
    temp_dir = tempfile.gettempdir()
    cleaned_count = 0
    cleaned_size = 0
    
    for pattern in TEMP_FILE_PATTERNS:
        for filepath in glob.glob(os.path.join(temp_dir, pattern)):
            try:
                file_age = current_time - os.path.getmtime(filepath)
                if force or file_age > TEMP_FILE_MAX_AGE_SECONDS:
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    cleaned_count += 1
                    cleaned_size += file_size
            except Exception as e:
                logger.debug(f"Could not clean {filepath}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"Cleaned {cleaned_count} temp files, freed {cleaned_size / (1024*1024):.2f} MB")

def run_garbage_collection():
    """Run garbage collection to free memory after heavy operations."""
    gc.collect()
    logger.debug("Garbage collection completed")

def cleanup_after_request():
    """Combined cleanup: temp files + garbage collection."""
    cleanup_temp_files()
    run_garbage_collection()

def safe_remove_file(filepath):
    """Safely remove a file with error handling."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
            return True
        except Exception as e:
            logger.debug(f"Could not remove {filepath}: {e}")
    return False

def safe_remove_files(*filepaths):
    """Safely remove multiple files."""
    for filepath in filepaths:
        safe_remove_file(filepath)

# =============================================================================
# HELPER FUNCTION: Centralized YouTube Audio Download
# This function handles all YouTube downloads with proper error handling,
# cookies support, and bypass techniques for 403 errors.
# =============================================================================
def download_audio_from_youtube(url: str, output_dir: str = None) -> str:
    """
    Download audio from YouTube URL using yt-dlp with robust error handling.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the file (uses temp dir if None)
    
    Returns:
        Path to the downloaded audio file
    
    Raises:
        Exception: If download fails
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "geo_bypass_country": "US",
        "socket_timeout": 1800,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
        "buffersize": 1024 * 16,
        "http_chunk_size": 10485760,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.5",
        },
    }
    
    # Add cookies file if available (critical for bypassing YouTube restrictions)
    if os.path.exists('cookies.txt'):
        ydl_opts['cookiefile'] = 'cookies.txt'
        logger.info("Using cookies.txt for YouTube authentication")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            logger.info(f"Downloaded YouTube audio to: {filename}")
            return filename
    except Exception as e:
        logger.error(f"Error downloading YouTube audio: {e}")
        raise

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.secret_key = os.environ.get("SESSION_SECRET") or "dev-secret-key-change-in-production"
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024

database_url = os.environ.get("DATABASE_URL") or os.environ.get("SUPABASE_URL")
if not database_url:
    raise RuntimeError(
        "DATABASE_URL or SUPABASE_URL environment variable is not set. "
        "Please set it to your PostgreSQL connection string. "
        "Example: postgresql://user:password@host:port/database"
    )
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'


@login_manager.user_loader
def load_user(user_id):
    return AdminUser.query.get(int(user_id))


@app.after_request
def after_request_cleanup(response):
    """Periodic cleanup after requests to manage memory and disk space."""
    cleanup_after_request()
    return response

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'wma'}

ANIME_SIMILARITY_THRESHOLD = 0.85
PODCAST_NAME_SIMILARITY_THRESHOLD = 90

# =============================================================================
# CENTRALIZED GEMINI API MANAGER
# =============================================================================
# Unified layer for all Gemini API calls with:
# - Support for up to 50 API keys (GEMINI_KEY_1 to GEMINI_KEY_50)
# - Smart round-robin distribution with cooldown
# - 2-key-max retry policy (protects API keys from exhaustion)
# - Session-based rate limiting (15 requests per 20 minutes)
# - Comprehensive logging with database persistence
# - Graceful quota exhaustion handling with Arabic error messages
# - Video transcription support up to 2 hours (120 minutes)
# - Admin dashboard with key management
# =============================================================================

class GeminiAPIManager:
    """Centralized Gemini API manager with intelligent key protection and rate limiting."""
    
    MAX_KEYS = 50
    RATE_LIMIT_REQUESTS = 15
    RATE_LIMIT_WINDOW_SECONDS = 1200
    KEY_COOLDOWN_SECONDS = 3600
    MAX_KEYS_PER_REQUEST = 2
    MAX_AUDIO_DURATION_MINUTES = 120
    MAX_LONG_VIDEO_PER_SESSION = 3
    LONG_VIDEO_THRESHOLD_MINUTES = 60
    MODEL_LIGHT = 'gemini-2.0-flash'
    MODEL_HEAVY = 'gemini-2.0-flash'
    
    def __init__(self):
        self.keys = self._load_keys()
        self.key_names = self._get_key_names()
        self.exhausted_keys = {}
        self.manually_disabled_keys = set()
        self.key_usage_count = defaultdict(int)
        self.key_last_used = {}
        self.key_cooldown_count = defaultdict(int)
        self.session_usage = defaultdict(list)
        self.session_long_video_count = defaultdict(int)
        self.current_key_index = 0
        self.hourly_stats = defaultdict(lambda: defaultdict(int))
        self.daily_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'quota_errors': 0,
            'other_errors': 0,
            'requests_by_operation': defaultdict(int),
            'requests_by_key': defaultdict(int)
        }
        self._lock = threading.Lock()
        self._db_initialized = False
        
        logging.info(f"[GeminiManager] Initialized with {len(self.keys)} API keys (max supported: {self.MAX_KEYS})")
        logging.info(f"[GeminiManager] Video transcription limit: {self.MAX_AUDIO_DURATION_MINUTES} minutes (2 hours)")
    
    def _load_keys(self):
        """Load all available Gemini API keys from environment (up to 50 keys)."""
        keys = []
        for i in range(1, self.MAX_KEYS + 1):
            key = os.environ.get(f"GEMINI_KEY_{i}")
            if key and key.strip():
                keys.append(key.strip())
        
        main_key = os.environ.get("GEMINI_API_KEY")
        if main_key and main_key.strip() and main_key.strip() not in keys:
            keys.append(main_key.strip())
        
        return keys
    
    def _get_key_names(self):
        """Get the environment variable names for each key."""
        names = {}
        key_index = 0
        for i in range(1, self.MAX_KEYS + 1):
            key = os.environ.get(f"GEMINI_KEY_{i}")
            if key and key.strip():
                names[key_index] = f"GEMINI_KEY_{i}"
                key_index += 1
        
        main_key = os.environ.get("GEMINI_API_KEY")
        if main_key and main_key.strip():
            found = False
            for idx, name in names.items():
                if self.keys[idx] == main_key.strip():
                    found = True
                    break
            if not found:
                names[len(names)] = "GEMINI_API_KEY"
        
        return names
    
    def init_db_state(self):
        """Initialize database state for keys - call after app context is available."""
        if self._db_initialized:
            return
        
        try:
            self._load_disabled_keys_from_db()
            self._db_initialized = True
            logging.info("[GeminiManager] Database state initialized")
        except Exception as e:
            logging.warning(f"[GeminiManager] Could not initialize DB state: {e}")
    
    def _load_disabled_keys_from_db(self):
        """Load manually disabled keys from database."""
        try:
            disabled_states = GeminiKeyState.query.filter_by(is_manually_disabled=True).all()
            for state in disabled_states:
                if state.key_index < len(self.keys):
                    self.manually_disabled_keys.add(state.key_index)
                    logging.info(f"[GeminiManager] Key {state.key_index + 1} loaded as manually disabled")
        except Exception as e:
            logging.warning(f"[GeminiManager] Could not load disabled keys: {e}")
    
    def _save_key_state_to_db(self, key_index, **updates):
        """Save key state to database."""
        try:
            key_name = self.key_names.get(key_index, f"KEY_{key_index + 1}")
            state = GeminiKeyState.query.filter_by(key_index=key_index).first()
            
            if not state:
                state = GeminiKeyState(key_index=key_index, key_name=key_name)
                db.session.add(state)
            
            for field, value in updates.items():
                if hasattr(state, field):
                    setattr(state, field, value)
            
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logging.warning(f"[GeminiManager] Could not save key state: {e}")
    
    def _log_usage_to_db(self, key_index, operation, is_retry, success, error_type=None, duration_minutes=None):
        """Log usage to database for dashboard analytics."""
        try:
            current_hour = datetime.utcnow().hour
            log_entry = GeminiUsageLog(
                key_index=key_index,
                operation_type=operation,
                is_retry=is_retry,
                success=success,
                error_type=error_type,
                content_duration_minutes=duration_minutes,
                session_id=self._get_session_id(),
                hour_bucket=current_hour
            )
            db.session.add(log_entry)
            
            today = date.today()
            daily = DailyStats.query.filter_by(date=today).first()
            if not daily:
                daily = DailyStats(date=today)
                db.session.add(daily)
            
            daily.total_requests += 1
            if success:
                daily.successful_requests += 1
            else:
                daily.failed_requests += 1
                if error_type == 'quota':
                    daily.quota_errors += 1
            
            if 'text' in operation.lower():
                daily.text_requests += 1
            elif 'vision' in operation.lower() or 'image' in operation.lower():
                daily.vision_requests += 1
            elif 'audio' in operation.lower() or 'transcription' in operation.lower():
                daily.audio_requests += 1
            elif 'video' in operation.lower():
                daily.video_requests += 1
            
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logging.debug(f"[GeminiManager] Could not log to DB: {e}")
    
    def _get_session_id(self):
        """Get current session/IP identifier for rate limiting."""
        try:
            if request:
                return request.remote_addr or 'unknown'
        except:
            pass
        return 'system'
    
    def _check_rate_limit(self, session_id):
        """Check if session has exceeded rate limit. Returns (allowed, remaining, reset_time)."""
        current_time = time.time()
        
        with self._lock:
            self.session_usage[session_id] = [
                t for t in self.session_usage[session_id]
                if current_time - t < self.RATE_LIMIT_WINDOW_SECONDS
            ]
            
            request_count = len(self.session_usage[session_id])
            remaining = max(0, self.RATE_LIMIT_REQUESTS - request_count)
            
            if request_count >= self.RATE_LIMIT_REQUESTS:
                oldest_request = min(self.session_usage[session_id])
                reset_time = int((oldest_request + self.RATE_LIMIT_WINDOW_SECONDS - current_time) / 60)
                return False, remaining, reset_time
            
            return True, remaining, 0
    
    def _record_session_request(self, session_id):
        """Record a request for rate limiting."""
        with self._lock:
            self.session_usage[session_id].append(time.time())
    
    def _get_available_key_index(self):
        """Get an available (non-exhausted, non-disabled) key index using round-robin."""
        current_time = time.time()
        
        with self._lock:
            expired_keys = [
                k for k, t in self.exhausted_keys.items()
                if current_time - t > self.KEY_COOLDOWN_SECONDS
            ]
            for k in expired_keys:
                del self.exhausted_keys[k]
                logging.info(f"[GeminiManager] Key {k + 1} cooldown expired, now available")
            
            available = [
                i for i in range(len(self.keys))
                if i not in self.exhausted_keys and i not in self.manually_disabled_keys
            ]
            
            if not available:
                return None
            
            start_index = self.current_key_index % len(self.keys)
            for offset in range(len(self.keys)):
                candidate = (start_index + offset) % len(self.keys)
                if candidate in available:
                    self.current_key_index = candidate + 1
                    self.key_usage_count[candidate] += 1
                    self.key_last_used[candidate] = current_time
                    return candidate
            
            selected = available[0]
            self.current_key_index = selected + 1
            self.key_usage_count[selected] += 1
            self.key_last_used[selected] = current_time
            return selected
    
    def _mark_key_exhausted(self, key_index):
        """Mark a key as exhausted with timestamp and update cooldown count."""
        with self._lock:
            self.exhausted_keys[key_index] = time.time()
            self.key_cooldown_count[key_index] += 1
            self.daily_stats['quota_errors'] += 1
            logging.warning(f"[GeminiManager] Key {key_index + 1} marked as exhausted (cooldown: 1 hour)")
        
        try:
            self._save_key_state_to_db(
                key_index,
                last_cooldown_at=datetime.utcnow(),
                cooldown_count=self.key_cooldown_count[key_index]
            )
        except:
            pass
    
    def set_key_disabled(self, key_index, disabled, reason=None):
        """Manually enable/disable a key."""
        with self._lock:
            if disabled:
                self.manually_disabled_keys.add(key_index)
                logging.info(f"[GeminiManager] Key {key_index + 1} manually disabled: {reason}")
            else:
                self.manually_disabled_keys.discard(key_index)
                logging.info(f"[GeminiManager] Key {key_index + 1} manually enabled")
        
        try:
            self._save_key_state_to_db(
                key_index,
                is_manually_disabled=disabled,
                disabled_at=datetime.utcnow() if disabled else None,
                disabled_reason=reason
            )
        except:
            pass
    
    def get_key_status(self, key_index):
        """Get the status of a specific key."""
        current_time = time.time()
        
        with self._lock:
            if key_index in self.manually_disabled_keys:
                return 'disabled'
            elif key_index in self.exhausted_keys:
                cooldown_remaining = self.KEY_COOLDOWN_SECONDS - (current_time - self.exhausted_keys[key_index])
                if cooldown_remaining > 0:
                    return 'cooldown'
                else:
                    del self.exhausted_keys[key_index]
                    return 'active'
            else:
                return 'active'
    
    def get_all_keys_info(self):
        """Get detailed info about all keys for dashboard."""
        current_time = time.time()
        keys_info = []
        
        with self._lock:
            for i in range(len(self.keys)):
                status = 'active'
                cooldown_remaining = 0
                
                if i in self.manually_disabled_keys:
                    status = 'disabled'
                elif i in self.exhausted_keys:
                    cooldown_remaining = self.KEY_COOLDOWN_SECONDS - (current_time - self.exhausted_keys[i])
                    if cooldown_remaining > 0:
                        status = 'cooldown'
                    else:
                        del self.exhausted_keys[i]
                
                keys_info.append({
                    'index': i,
                    'name': self.key_names.get(i, f"KEY_{i + 1}"),
                    'display_name': f"Key #{i + 1}",
                    'status': status,
                    'usage_count': self.key_usage_count.get(i, 0),
                    'cooldown_count': self.key_cooldown_count.get(i, 0),
                    'last_used': self.key_last_used.get(i),
                    'cooldown_remaining_minutes': max(0, int(cooldown_remaining / 60)) if status == 'cooldown' else 0
                })
        
        return keys_info
    
    def get_dashboard_stats(self):
        """Get comprehensive stats for admin dashboard."""
        current_time = time.time()
        
        with self._lock:
            active_count = len([
                i for i in range(len(self.keys))
                if i not in self.exhausted_keys and i not in self.manually_disabled_keys
            ])
            cooldown_count = len([
                i for i in self.exhausted_keys
                if current_time - self.exhausted_keys[i] < self.KEY_COOLDOWN_SECONDS
            ])
            disabled_count = len(self.manually_disabled_keys)
            
            return {
                'total_keys': len(self.keys),
                'active_keys': active_count,
                'cooldown_keys': cooldown_count,
                'disabled_keys': disabled_count,
                'total_requests_today': self.daily_stats['total_requests'],
                'successful_requests_today': self.daily_stats['successful_requests'],
                'quota_errors_today': self.daily_stats['quota_errors'],
                'other_errors_today': self.daily_stats['other_errors'],
                'requests_by_operation': dict(self.daily_stats['requests_by_operation']),
                'requests_by_key': dict(self.daily_stats['requests_by_key'])
            }
    
    def check_long_video_limit(self, session_id, duration_minutes):
        """Check if session has exceeded long video transcription limit."""
        if duration_minutes < self.LONG_VIDEO_THRESHOLD_MINUTES:
            return True, None
        
        with self._lock:
            current_time = time.time()
            window_start = current_time - self.RATE_LIMIT_WINDOW_SECONDS
            
            if self.session_long_video_count.get(session_id, 0) >= self.MAX_LONG_VIDEO_PER_SESSION:
                return False, (
                    f"لقد وصلت إلى الحد الأقصى من عمليات تحويل الفيديو الطويل "
                    f"({self.MAX_LONG_VIDEO_PER_SESSION} عمليات). "
                    f"الرجاء الانتظار قليلاً قبل المحاولة مرة أخرى."
                )
            
            return True, None
    
    def record_long_video(self, session_id):
        """Record a long video transcription for rate limiting."""
        with self._lock:
            self.session_long_video_count[session_id] = self.session_long_video_count.get(session_id, 0) + 1
    
    def _is_quota_error(self, error_str):
        """Check if error is related to quota/rate limiting."""
        quota_indicators = ['429', 'quota', 'rate', 'limit', 'exhausted', 'resource_exhausted']
        return any(ind in error_str for ind in quota_indicators)
    
    def _is_auth_error(self, error_str):
        """Check if error is related to authentication/authorization."""
        auth_indicators = ['400', '401', '403', 'invalid', 'unauthorized', 'permission', 'api_key']
        return any(ind in error_str for ind in auth_indicators)
    
    def _log_request(self, operation, key_index, is_retry, success, error_type=None, duration_minutes=None):
        """Log API request for monitoring."""
        current_hour = datetime.utcnow().hour
        
        with self._lock:
            self.daily_stats['total_requests'] += 1
            self.daily_stats['requests_by_operation'][operation] += 1
            self.daily_stats['requests_by_key'][key_index + 1] += 1
            self.hourly_stats[current_hour]['total'] += 1
            self.hourly_stats[current_hour][operation] += 1
            
            if success:
                self.daily_stats['successful_requests'] += 1
                self.hourly_stats[current_hour]['success'] += 1
            elif error_type:
                if 'quota' in error_type:
                    self.daily_stats['quota_errors'] += 1
                    self.hourly_stats[current_hour]['quota_errors'] += 1
                else:
                    self.daily_stats['other_errors'] += 1
                    self.hourly_stats[current_hour]['other_errors'] += 1
        
        retry_str = " (retry)" if is_retry else ""
        status = "SUCCESS" if success else f"FAILED ({error_type})"
        duration_str = f" [{duration_minutes:.1f} min]" if duration_minutes else ""
        logging.info(f"[GeminiManager] [{operation}] Key {key_index + 1}{retry_str}{duration_str} - {status}")
        
        try:
            self._log_usage_to_db(key_index, operation, is_retry, success, error_type, duration_minutes)
        except:
            pass
    
    def get_hourly_stats(self):
        """Get hourly statistics for the last 24 hours."""
        with self._lock:
            return dict(self.hourly_stats)
    
    def get_stats(self):
        """Get current API usage statistics."""
        with self._lock:
            available_keys = len(self.keys) - len(self.exhausted_keys)
            return {
                'total_keys': len(self.keys),
                'available_keys': available_keys,
                'exhausted_keys': len(self.exhausted_keys),
                **dict(self.daily_stats)
            }
    
    def get_available_keys_count(self):
        """Get count of currently available (non-exhausted) keys."""
        current_time = time.time()
        with self._lock:
            available = len([
                k for k in range(len(self.keys))
                if k not in self.exhausted_keys or 
                current_time - self.exhausted_keys[k] > self.KEY_COOLDOWN_SECONDS
            ])
            return available
    
    def get_time_until_key_available(self):
        """Get seconds until earliest exhausted key becomes available."""
        with self._lock:
            if not self.exhausted_keys:
                return 0
            current_time = time.time()
            earliest = min(self.exhausted_keys.values())
            remaining = self.KEY_COOLDOWN_SECONDS - (current_time - earliest)
            return max(0, remaining)
    
    def has_keys(self):
        """Check if any API keys are configured."""
        return len(self.keys) > 0
    
    def call_text(self, prompt, operation='text', use_heavy_model=False, session_id=None):
        """
        Call Gemini for text prompts with 2-key-max retry policy.
        
        Args:
            prompt: Text prompt
            operation: Operation name for logging (e.g., 'podcast_identify', 'anime_search')
            use_heavy_model: Whether to use the heavier model (for complex tasks)
            session_id: Optional session ID for rate limiting
        
        Returns:
            str or None: Response text or None on failure
        
        Raises:
            ValueError: If rate limit exceeded or quota exhausted with Arabic message
        """
        if not self.keys:
            logging.warning("[GeminiManager] No API keys configured")
            return None
        
        if session_id is None:
            session_id = self._get_session_id()
        
        allowed, remaining, reset_time = self._check_rate_limit(session_id)
        if not allowed:
            raise ValueError(
                f"تم بلوغ الحد المسموح به من الطلبات. الرجاء الانتظار {reset_time} دقيقة."
            )
        
        model_name = self.MODEL_HEAVY if use_heavy_model else self.MODEL_LIGHT
        keys_tried = 0
        last_error = None
        
        for attempt in range(self.MAX_KEYS_PER_REQUEST):
            key_index = self._get_available_key_index()
            
            if key_index is None:
                time_remaining = int(self.get_time_until_key_available() / 60)
                raise ValueError(
                    f"تم استهلاك الحد المتاح من خدمة الذكاء الاصطناعي مؤقتاً. "
                    f"الرجاء المحاولة بعد {time_remaining} دقيقة أو تجربة أداة أخرى."
                )
            
            keys_tried += 1
            is_retry = attempt > 0
            
            try:
                genai.configure(api_key=self.keys[key_index])
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                
                if response and response.text:
                    self._record_session_request(session_id)
                    self._log_request(operation, key_index, is_retry, success=True)
                    return response.text.strip()
                
                self._log_request(operation, key_index, is_retry, success=False, error_type='empty_response')
                return None
                
            except Exception as e:
                error_str = str(e).lower()
                last_error = str(e)
                
                if self._is_quota_error(error_str):
                    self._mark_key_exhausted(key_index)
                    self._log_request(operation, key_index, is_retry, success=False, error_type='quota')
                    continue
                elif self._is_auth_error(error_str):
                    self._log_request(operation, key_index, is_retry, success=False, error_type='auth')
                    logging.error(f"[GeminiManager] Auth error on key {key_index + 1}: {e}")
                    return None
                else:
                    self._log_request(operation, key_index, is_retry, success=False, error_type='other')
                    logging.error(f"[GeminiManager] API error: {e}")
                    return None
        
        logging.warning(f"[GeminiManager] All {keys_tried} keys exhausted for {operation}")
        raise ValueError(
            "خدمة الذكاء الاصطناعي مشغولة حالياً. الرجاء المحاولة لاحقاً."
        )
    
    def call_vision(self, image_or_path, prompt, operation='vision', use_heavy_model=False, session_id=None):
        """
        Call Gemini Vision with 2-key-max retry policy.
        
        Args:
            image_or_path: PIL Image object or path to image file
            prompt: Text prompt
            operation: Operation name for logging
            use_heavy_model: Whether to use the heavier model
            session_id: Optional session ID for rate limiting
        
        Returns:
            str or None: Response text or None on failure
        
        Raises:
            ValueError: If rate limit exceeded or quota exhausted
        """
        if not self.keys:
            logging.warning("[GeminiManager] No API keys configured")
            return None
        
        if session_id is None:
            session_id = self._get_session_id()
        
        allowed, remaining, reset_time = self._check_rate_limit(session_id)
        if not allowed:
            raise ValueError(
                f"تم بلوغ الحد المسموح به من الطلبات. الرجاء الانتظار {reset_time} دقيقة."
            )
        
        if isinstance(image_or_path, str):
            img = Image.open(image_or_path)
        else:
            img = image_or_path
        
        model_name = self.MODEL_HEAVY if use_heavy_model else self.MODEL_LIGHT
        keys_tried = 0
        last_error = None
        
        for attempt in range(self.MAX_KEYS_PER_REQUEST):
            key_index = self._get_available_key_index()
            
            if key_index is None:
                time_remaining = int(self.get_time_until_key_available() / 60)
                raise ValueError(
                    f"تم استهلاك الحد المتاح من خدمة الذكاء الاصطناعي مؤقتاً. "
                    f"الرجاء المحاولة بعد {time_remaining} دقيقة أو تجربة أداة أخرى."
                )
            
            keys_tried += 1
            is_retry = attempt > 0
            
            try:
                genai.configure(api_key=self.keys[key_index])
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, img])
                
                if response and response.text:
                    self._record_session_request(session_id)
                    self._log_request(operation, key_index, is_retry, success=True)
                    return response.text.strip()
                
                self._log_request(operation, key_index, is_retry, success=False, error_type='empty_response')
                return None
                
            except Exception as e:
                error_str = str(e).lower()
                last_error = str(e)
                
                if self._is_quota_error(error_str):
                    self._mark_key_exhausted(key_index)
                    self._log_request(operation, key_index, is_retry, success=False, error_type='quota')
                    continue
                elif self._is_auth_error(error_str):
                    self._log_request(operation, key_index, is_retry, success=False, error_type='auth')
                    logging.error(f"[GeminiManager] Auth error on key {key_index + 1}: {e}")
                    return None
                else:
                    self._log_request(operation, key_index, is_retry, success=False, error_type='other')
                    logging.error(f"[GeminiManager] Vision API error: {e}")
                    return None
        
        logging.warning(f"[GeminiManager] All {keys_tried} keys exhausted for {operation}")
        raise ValueError(
            "خدمة الذكاء الاصطناعي مشغولة حالياً. الرجاء المحاولة لاحقاً."
        )
    
    def call_audio_transcription(self, audio_path, prompt, operation='transcription', session_id=None):
        """
        Call Gemini for audio transcription with file upload and 2-key-max retry policy.
        
        Args:
            audio_path: Path to audio file
            prompt: Transcription prompt
            operation: Operation name for logging
            session_id: Optional session ID for rate limiting
        
        Returns:
            str or None: Transcription text or None on failure
        
        Raises:
            ValueError: If rate limit exceeded, quota exhausted, or file too large
        """
        if not self.keys:
            raise ValueError(
                "لم يتم إعداد مفاتيح Gemini API. يرجى إضافة GEMINI_API_KEY في الإعدادات."
            )
        
        if session_id is None:
            session_id = self._get_session_id()
        
        allowed, remaining, reset_time = self._check_rate_limit(session_id)
        if not allowed:
            raise ValueError(
                f"تم بلوغ الحد المسموح به من الطلبات. الرجاء الانتظار {reset_time} دقيقة."
            )
        
        keys_tried = 0
        last_error = None
        
        for attempt in range(self.MAX_KEYS_PER_REQUEST):
            key_index = self._get_available_key_index()
            
            if key_index is None:
                time_remaining = int(self.get_time_until_key_available() / 60)
                raise ValueError(
                    f"تم استنفاد حصة جميع مفاتيح API. الرجاء الانتظار {time_remaining} دقيقة أو إضافة مفاتيح جديدة."
                )
            
            keys_tried += 1
            is_retry = attempt > 0
            audio_file = None
            
            try:
                genai.configure(api_key=self.keys[key_index])
                
                logging.info(f"[GeminiManager] [{operation}] Key {key_index + 1} - Uploading audio...")
                audio_file = genai.upload_file(audio_path)
                
                model = genai.GenerativeModel(self.MODEL_LIGHT)
                logging.info(f"[GeminiManager] [{operation}] Key {key_index + 1} - Transcribing (timeout: 30 min)...")
                
                response = model.generate_content(
                    [prompt, audio_file],
                    request_options={"timeout": 1800}
                )
                
                try:
                    genai.delete_file(audio_file.name)
                except:
                    pass
                
                if response and response.text:
                    self._record_session_request(session_id)
                    self._log_request(operation, key_index, is_retry, success=True)
                    text = response.text.strip()
                    word_count = len(text.split())
                    logging.info(f"[GeminiManager] [{operation}] Success: {len(text)} chars, ~{word_count} words")
                    return text
                
                self._log_request(operation, key_index, is_retry, success=False, error_type='empty_response')
                last_error = "Empty response from API"
                continue
                
            except Exception as e:
                error_str = str(e).lower()
                last_error = str(e)
                
                try:
                    if audio_file:
                        genai.delete_file(audio_file.name)
                except:
                    pass
                
                if self._is_quota_error(error_str):
                    self._mark_key_exhausted(key_index)
                    self._log_request(operation, key_index, is_retry, success=False, error_type='quota')
                    continue
                elif 'timeout' in error_str or 'deadline' in error_str:
                    self._log_request(operation, key_index, is_retry, success=False, error_type='timeout')
                    logging.warning(f"[GeminiManager] Timeout on key {key_index + 1}")
                    continue
                elif self._is_auth_error(error_str):
                    self._log_request(operation, key_index, is_retry, success=False, error_type='auth')
                    logging.error(f"[GeminiManager] Invalid API key {key_index + 1}")
                    return None
                else:
                    self._log_request(operation, key_index, is_retry, success=False, error_type='other')
                    logging.error(f"[GeminiManager] Transcription error: {e}")
                    continue
        
        logging.warning(f"[GeminiManager] All {keys_tried} keys exhausted for {operation}")
        raise ValueError(
            f"تم استنفاد حصة جميع مفاتيح API. حاول مرة أخرى بعد ساعة أو أضف مفاتيح جديدة."
        )
    
    def check_audio_duration_limit(self, audio_path):
        """
        Check if audio/video duration is within allowed limits (2 hours max).
        
        Returns:
            tuple: (allowed: bool, duration_minutes: float, error_message: str or None)
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            duration_minutes = len(audio) / 60000
            
            if duration_minutes > self.MAX_AUDIO_DURATION_MINUTES:
                hours = duration_minutes / 60
                max_hours = self.MAX_AUDIO_DURATION_MINUTES / 60
                return False, duration_minutes, (
                    f"مدة الملف ({hours:.1f} ساعة / {duration_minutes:.0f} دقيقة) أكبر من الحد الأقصى المسموح "
                    f"({max_hours:.0f} ساعة). يُنصح بتقسيم الملف إلى أجزاء أقصر."
                )
            
            return True, duration_minutes, None
            
        except Exception as e:
            logging.error(f"[GeminiManager] Could not check audio duration: {e}")
            return True, 0, None


gemini_manager = GeminiAPIManager()

GEMINI_KEYS = gemini_manager.keys


def get_available_keys_count():
    """Get count of currently available (non-exhausted) keys - backward compatibility."""
    return gemini_manager.get_available_keys_count()


def get_time_until_key_available():
    """Get seconds until earliest key available - backward compatibility."""
    return gemini_manager.get_time_until_key_available()


def call_gemini_text(prompt, max_retries=None, operation='text'):
    """
    Call Gemini API for text-only prompts - backward compatible wrapper.
    Now uses 2-key-max retry policy instead of rotating through all keys.
    
    Args:
        prompt: The text prompt to send to the model
        max_retries: Ignored (kept for backward compatibility)
        operation: Operation name for logging
    
    Returns:
        str: Response text or None if all keys exhausted
    """
    try:
        return gemini_manager.call_text(prompt, operation=operation)
    except ValueError as e:
        logging.warning(f"[call_gemini_text] {e}")
        return None


def call_gemini_vision(image_or_path, prompt, max_retries=None, operation='vision'):
    """
    Call Gemini Vision API - backward compatible wrapper.
    Now uses 2-key-max retry policy instead of rotating through all keys.
    
    Args:
        image_or_path: Either a PIL Image object or path to image file
        prompt: The prompt to send to the model
        max_retries: Ignored (kept for backward compatibility)
        operation: Operation name for logging
    
    Returns:
        str: Response text or None if all keys exhausted
    """
    try:
        return gemini_manager.call_vision(image_or_path, prompt, operation=operation)
    except ValueError as e:
        logging.warning(f"[call_gemini_vision] {e}")
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
    Use Gemini to transcribe audio file with centralized API management.
    
    Features:
    - Uses centralized GeminiAPIManager with 2-key-max retry policy
    - Duration limit: 15 minutes max (protects API quota)
    - Session-based rate limiting
    - Comprehensive logging for monitoring
    - Graceful error handling with Arabic messages
    """
    allowed, duration_minutes, error_msg = gemini_manager.check_audio_duration_limit(audio_path)
    if not allowed:
        raise ValueError(error_msg)
    
    logging.info(f"[Transcription] Audio duration: {duration_minutes:.1f} minutes")
    
    compressed_path = compress_audio_for_upload(audio_path)
    upload_path = compressed_path if compressed_path != audio_path else audio_path
    
    language_name = LANGUAGE_NAMES.get(language, 'Arabic')
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
- Maximum output: 22,000 words - use all of it if needed"""
    
    try:
        result = gemini_manager.call_audio_transcription(
            upload_path, 
            prompt, 
            operation='audio_transcription'
        )
        
        if compressed_path != audio_path:
            safe_remove_file(compressed_path)
        
        return result
        
    except ValueError:
        if compressed_path != audio_path:
            safe_remove_file(compressed_path)
        raise
    except Exception as e:
        if compressed_path != audio_path:
            safe_remove_file(compressed_path)
        logging.error(f"[Transcription] Unexpected error: {e}")
        raise ValueError(f"فشل في تحويل الصوت إلى نص. الخطأ: {str(e)[:100]}")


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
    return render_template('index.html',
                         supabase_url=os.environ.get('SUPABASE_URL', ''),
                         supabase_anon_key=os.environ.get('SUPABASE_ANON_KEY', ''))


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
            'socket_timeout': 1800,
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


# =============================================================================
# دالة جلب الجودات المتاحة من يوتيوب
# =============================================================================
@app.route('/get-video-formats', methods=['POST'])
def get_video_formats():
    """جلب قائمة الجودات والفورمات المتاحة للفيديو"""
    try:
        data = request.get_json()
        url = data.get('url', '')

        if not url:
            return jsonify({'error': 'الرجاء إدخال رابط الفيديو'}), 400

        logging.info(f"جلب الجودات المتاحة للفيديو: {url}")

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'noplaylist': True,
            'socket_timeout': 1800,
            'retries': 3,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
        }

        if os.path.exists('cookies.txt'):
            ydl_opts['cookiefile'] = 'cookies.txt'

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if not info:
                return jsonify({'error': 'لم يتم العثور على معلومات الفيديو'}), 400

            formats = info.get('formats', [])
            available_qualities = []
            seen_resolutions = set()

            # فلترة وتجميع الجودات المتاحة
            for fmt in formats:
                height = fmt.get('height')
                vcodec = fmt.get('vcodec', 'none')
                acodec = fmt.get('acodec', 'none')
                ext = fmt.get('ext', '')
                filesize = fmt.get('filesize') or fmt.get('filesize_approx') or 0

                # تجاهل الفورمات بدون ارتفاع محدد
                if height and vcodec != 'none':
                    resolution = f"{height}p"
                    if resolution not in seen_resolutions:
                        seen_resolutions.add(resolution)
                        size_mb = round(filesize / (1024 * 1024), 1) if filesize else 0
                        available_qualities.append({
                            'resolution': resolution,
                            'height': height,
                            'type': 'video',
                            'size_mb': size_mb,
                            'format': ext
                        })

            # ترتيب الجودات من الأعلى للأقل
            available_qualities.sort(key=lambda x: x['height'], reverse=True)

            # إضافة خيار الصوت فقط
            audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
            best_audio = None
            for af in audio_formats:
                if af.get('abr'):
                    if not best_audio or af.get('abr', 0) > best_audio.get('abr', 0):
                        best_audio = af

            # جلب معلومات الفيديو الأساسية
            duration_seconds = info.get('duration') or 0
            title = info.get('title') or 'غير معروف'
            thumbnail = info.get('thumbnail') or ''

            logging.info(f"تم جلب {len(available_qualities)} جودة متاحة")

            return jsonify({
                'success': True,
                'title': title,
                'duration_seconds': duration_seconds,
                'thumbnail': thumbnail,
                'qualities': available_qualities,
                'has_audio': best_audio is not None,
                'default_quality': available_qualities[0]['resolution'] if available_qualities else '720p'
            })

    except Exception as e:
        logging.error(f"خطأ في جلب الجودات: {str(e)}")
        return jsonify({'error': f'خطأ في جلب الجودات المتاحة: {str(e)}'}), 400


# =============================================================================
# دالة تحميل الفيديو/الصوت من يوتيوب
# =============================================================================
def download_youtube_media(url: str, quality: str, download_type: str, output_dir: str) -> str:
    """
    تحميل الفيديو أو الصوت من يوتيوب بالجودة المحددة
    
    Args:
        url: رابط الفيديو
        quality: الجودة المطلوبة (مثل 720p, 1080p)
        download_type: نوع التحميل (video أو audio)
        output_dir: مجلد الحفظ
    
    Returns:
        مسار الملف المحمل
    """
    unique_id = str(uuid.uuid4())[:8]
    
    if download_type == 'audio':
        # تحميل الصوت فقط
        output_template = os.path.join(output_dir, f'audio_{unique_id}.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 7200,
            'retries': 5,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            },
        }
    else:
        # تحميل الفيديو بالجودة المحددة
        height = quality.replace('p', '') if quality else '720'
        output_template = os.path.join(output_dir, f'video_{unique_id}.%(ext)s')
        
        # اختيار أفضل فورمات بناءً على الجودة المطلوبة
        format_string = f'bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={height}]+bestaudio/best[height<={height}]/best'
        
        ydl_opts = {
            'format': format_string,
            'merge_output_format': 'mp4',
            'outtmpl': output_template,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'socket_timeout': 7200,
            'retries': 5,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            },
        }

    if os.path.exists('cookies.txt'):
        ydl_opts['cookiefile'] = 'cookies.txt'

    logging.info(f"بدء تحميل {download_type} بجودة {quality}")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # البحث عن الملف المحمل
    base_name = output_template.replace('.%(ext)s', '')
    for ext in ['mp4', 'mp3', 'mkv', 'webm', 'm4a', 'wav']:
        possible_file = f"{base_name}.{ext}"
        if os.path.exists(possible_file):
            logging.info(f"تم تحميل الملف: {possible_file}")
            return possible_file

    raise Exception("لم يتم العثور على الملف المحمل")


# =============================================================================
# دالة قص الملف باستخدام ffmpeg
# =============================================================================
def cut_media_segment(input_path: str, start_seconds: int, end_seconds: int, 
                      download_type: str, output_dir: str) -> str:
    """
    قص جزء من الفيديو أو الصوت باستخدام ffmpeg
    
    Args:
        input_path: مسار الملف الأصلي
        start_seconds: وقت البداية بالثواني
        end_seconds: وقت النهاية بالثواني
        download_type: نوع الملف (video أو audio)
        output_dir: مجلد الحفظ
    
    Returns:
        مسار الملف المقصوص
    """
    unique_id = str(uuid.uuid4())[:8]
    duration = end_seconds - start_seconds
    
    if download_type == 'audio':
        output_path = os.path.join(output_dir, f'clip_{unique_id}.mp3')
        # قص الصوت مع ترميز خفيف
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_seconds),
            '-i', input_path,
            '-t', str(duration),
            '-c:a', 'libmp3lame',
            '-b:a', '192k',
            '-ar', '44100',
            output_path
        ]
    else:
        output_path = os.path.join(output_dir, f'clip_{unique_id}.mp4')
        # قص الفيديو بدون إعادة ترميز قدر الإمكان (أسرع وأخف)
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_seconds),
            '-i', input_path,
            '-t', str(duration),
            '-c', 'copy',  # نسخ مباشر بدون إعادة ترميز
            '-avoid_negative_ts', 'make_zero',
            output_path
        ]

    logging.info(f"بدء قص الملف من {start_seconds}s إلى {end_seconds}s")
    
    try:
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=7200)
        
        # إذا فشل القص بدون ترميز، نحاول مع إعادة الترميز
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            if download_type == 'video':
                logging.info("القص بدون ترميز فشل، جاري المحاولة مع إعادة الترميز...")
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_seconds),
                    '-i', input_path,
                    '-t', str(duration),
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-strict', 'experimental',
                    output_path
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=7200)
    except subprocess.CalledProcessError as e:
        # محاولة أخيرة مع إعادة الترميز الكامل
        if download_type == 'video':
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_seconds),
                '-i', input_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', 'experimental',
                output_path
            ]
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, timeout=7200)
        else:
            raise e

    if os.path.exists(output_path):
        logging.info(f"تم قص الملف بنجاح: {output_path}")
        return output_path
    else:
        raise Exception("فشل في إنشاء الملف المقصوص")


@app.route('/process-video', methods=['POST'])
def process_video():
    """
    معالجة وقص الفيديو من يوتيوب مع دعم اختيار الجودة ونوع التحميل
    الحقول المدعومة:
    - url: رابط الفيديو (مطلوب)
    - start_time: وقت البداية (مطلوب)
    - end_time: وقت النهاية (مطلوب)
    - quality: الجودة (اختياري - افتراضي 720p)
    - download_type: نوع التحميل video أو audio (اختياري - افتراضي video)
    """
    temp_media = None
    output_file = None

    try:
        data = request.get_json()
        url = data.get('url', '')
        start_time = data.get('start_time', '00:00')
        end_time = data.get('end_time', '')
        quality = data.get('quality', '720p')  # الجودة الافتراضية
        download_type = data.get('download_type', 'video')  # النوع الافتراضي

        logging.info(f"طلب قص فيديو: quality={quality}, type={download_type}")

        if not url:
            return jsonify({'error': 'الرجاء إدخال رابط الفيديو'}), 400

        if not end_time:
            return jsonify({'error': 'الرجاء تحديد وقت النهاية'}), 400

        # التحقق من نوع التحميل
        if download_type not in ['video', 'audio']:
            download_type = 'video'

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

        # تحميل الملف باستخدام الدالة المساعدة الجديدة
        logging.info(f"بدء تحميل الملف من يوتيوب...")
        temp_media = download_youtube_media(url, quality, download_type, UPLOAD_FOLDER)
        logging.info(f"تم تحميل الملف: {temp_media}")

        # قص الملف باستخدام الدالة المساعدة
        logging.info(f"بدء قص الملف...")
        output_file = cut_media_segment(temp_media, start_seconds, end_seconds, 
                                         download_type, UPLOAD_FOLDER)
        logging.info(f"تم قص الملف: {output_file}")

        # حذف الملف الأصلي بعد القص
        safe_remove_file(temp_media)
        logging.info(f"تم حذف الملف المؤقت: {temp_media}")
        temp_media = None

        # تنظيف الملف بعد الإرسال
        @after_this_request
        def cleanup(response):
            if safe_remove_file(output_file):
                logging.info(f"تم حذف ملف المخرجات: {output_file}")
            return response

        # إرجاع الملف حسب النوع
        if download_type == 'audio':
            download_name = f'clip_{start_time.replace(":", "-")}_{end_time.replace(":", "-")}.mp3'
            mimetype = 'audio/mpeg'
        else:
            download_name = f'clip_{start_time.replace(":", "-")}_{end_time.replace(":", "-")}_{quality}.mp4'
            mimetype = 'video/mp4'

        return send_file(
            output_file,
            as_attachment=True,
            download_name=download_name,
            mimetype=mimetype)

    except subprocess.CalledProcessError as e:
        logging.error(f"خطأ في معالجة الفيديو: {e}")
        safe_remove_files(temp_media, output_file)
        return jsonify({
            'error':
            f'خطأ في معالجة الفيديو: {e.stderr.decode() if e.stderr else str(e)}'
        }), 500
    except Exception as e:
        logging.error(f"خطأ عام: {str(e)}")
        safe_remove_files(temp_media, output_file)
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
            safe_remove_file(temp_path)

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
            safe_remove_file(temp_path)
            return jsonify({
                'found': False,
                'message':
                'فشل الاتصال بخدمة البحث. تأكد من إعداد مفاتيح Gemini API أو جرب لاحقاً.',
                'suggest_search_by_name': True
            })

        data = response.json()

        if data.get('error'):
            safe_remove_file(temp_path)
            return jsonify({'error': data['error']}), 400

        results = data.get('result', [])

        if not results:
            safe_remove_file(temp_path)
            return jsonify({
                'found': False,
                'message':
                'لم يتم العثور على نتائج. جرب استخدام صورة أوضح أو البحث بالاسم.',
                'suggest_search_by_name': True
            })

        top_result = results[0]
        similarity = top_result.get('similarity', 0)

        if similarity < ANIME_SIMILARITY_THRESHOLD:
            safe_remove_file(temp_path)
            return jsonify({
                'found': False,
                'message':
                'لم يتم العثور على تطابق دقيق. الرجاء استخدام صورة أوضح أو جرب البحث بالاسم.',
                'suggest_search_by_name': True
            })

        safe_remove_file(temp_path)

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
        safe_remove_file(temp_path)
        return jsonify(
            {'error': 'انتهت مهلة الاتصال. الرجاء المحاولة مرة أخرى.'}), 500
    except Exception as e:
        safe_remove_file(temp_path)
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
            safe_remove_file(temp_audio)
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
                safe_remove_file(temp_audio)
                run_garbage_collection()
                return jsonify({'error': str(ve)}), 400
            except Exception as e:
                logging.error(f"Gemini transcription failed: {str(e)}")
                text = None

            if text:
                safe_remove_file(temp_audio)
                run_garbage_collection()

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
                safe_remove_file(temp_audio)
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
                            if chunk_audio:
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

        safe_remove_files(temp_audio, temp_wav, *chunk_files)
        run_garbage_collection()

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
        safe_remove_files(temp_audio, temp_wav, *chunk_files)
        run_garbage_collection()
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        error_message = str(e)
        logging.error(f"Transcription error: {error_message}")
        safe_remove_files(temp_audio, temp_wav, *chunk_files)
        run_garbage_collection()

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

        safe_remove_file(temp_image)

        if not text.strip():
            return jsonify({
                'success': True,
                'text': '',
                'message': 'لم يتم العثور على نص في الصورة'
            })

        return jsonify({'success': True, 'text': text.strip()})

    except Exception as e:
        safe_remove_file(temp_image)
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
            safe_remove_file(temp_image)
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
            safe_remove_file(temp_image)
            return jsonify({'error': 'فشل الاتصال بخدمة البحث'}), 500

        itunes_data = itunes_response.json()
        podcasts = itunes_data.get('results', [])

        smart_links = generate_podcast_search_links(search_term)

        if not podcasts:
            safe_remove_file(temp_image)
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
                safe_remove_file(temp_image)
                return jsonify({
                    'found': False,
                    'extracted_text': extracted_text,
                    'detection_method': detection_method,
                    'similarity': best_similarity,
                    'message':
                    'تعذر التعرف على البودكاست بدقة عالية. جرب الروابط أدناه للبحث يدوياً.',
                    'search_links': smart_links
                })

        safe_remove_file(temp_image)

        if not best_match:
            return jsonify({
                'found': False,
                'message': 'لم يتم العثور على بودكاست مطابق',
                'search_links': smart_links
            })

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
        safe_remove_file(temp_image)
        return jsonify(
            {'error': 'انتهت مهلة الاتصال. الرجاء المحاولة مرة أخرى.'}), 500
    except Exception as e:
        safe_remove_file(temp_image)
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

        safe_remove_files(temp_audio, temp_wav)

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
        safe_remove_files(temp_audio, temp_wav)
        return jsonify({'error': f'خطأ: {str(e)}'}), 500


def cleanup_download_files(output_template, output_file):
    """Clean up all possible temp files from download"""
    for ext in [
            'mp4', 'mkv', 'webm', 'avi', 'mov', 'mp3', 'wav', 'ogg', 'm4a',
            'part', 'ytdl', 'temp'
    ]:
        temp_file = f'{output_template}.{ext}'
        safe_remove_file(temp_file)
    safe_remove_file(output_file)


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
            'socket_timeout': 1800,
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
                                safe_remove_file(alt_file)
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
                                if alt_file != output_file:
                                    safe_remove_file(alt_file)
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
                        safe_remove_file(docx_path)
                        use_ocr = True
                    else:
                        conversion_success = True
                        use_ocr = False

            except Exception as e:
                logging.warning(f"Standard PDF conversion failed: {e}")
                if extraction_method == 'standard':
                    safe_remove_file(pdf_path)
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
                    safe_remove_file(pdf_path)
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
                safe_remove_files(pdf_path, docx_path)
                return jsonify({'error':
                                f'فشل في الترجمة: {str(e)[:100]}'}), 500

        if not os.path.exists(docx_path):
            safe_remove_file(pdf_path)
            return jsonify({'error': 'فشل في إنشاء ملف Word'}), 500

        final_pdf = pdf_path
        final_docx = docx_path

        @after_this_request
        def cleanup(response):
            safe_remove_files(final_pdf, final_docx)
            return response

        return send_file(
            docx_path,
            as_attachment=True,
            download_name='converted_document.docx',
            mimetype=
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )

    except Exception as e:
        safe_remove_files(pdf_path, docx_path)
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
                1800,
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
                                safe_remove_file(alt_file)
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
                safe_remove_file(temp_video)
                return jsonify({
                    'error':
                    'فشل في استخراج الصوت من الفيديو. تأكد من أن الفيديو يحتوي على صوت.'
                }), 400

            safe_remove_file(temp_video)
            temp_video = None

        logging.info(
            f"Starting Gemini transcription for video audio: {temp_audio} with language: {language}"
        )

        try:
            text = transcribe_audio_with_gemini(temp_audio, language)
        except ValueError as ve:
            safe_remove_file(temp_audio)
            run_garbage_collection()
            return jsonify({'error': str(ve)}), 400
        except Exception as transcribe_error:
            logging.error(f"Transcription error: {transcribe_error}")
            safe_remove_files(temp_audio, temp_video)
            run_garbage_collection()
            return jsonify({
                'error':
                'فشل في تحويل الصوت إلى نص. حاول مرة أخرى أو استخدم ملف أقصر.'
            }), 500

        safe_remove_file(temp_audio)
        run_garbage_collection()

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
        safe_remove_files(temp_video, temp_audio)
        run_garbage_collection()
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logging.error(f"Video transcription error: {e}")
        safe_remove_files(temp_video, temp_audio)
        run_garbage_collection()
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


def apply_basic_format(doc: Document):
    """
    Basic formatting: margins 2.5 cm, font size 14, justified paragraphs, 1.5 line spacing.
    """
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    try:
        normal_style = doc.styles["Normal"]
        normal_style.font.name = "Times New Roman"
        normal_style.font.size = Pt(14)
    except Exception:
        pass

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        pf = para.paragraph_format
        pf.line_spacing = 1.5
        pf.space_before = Pt(0)
        pf.space_after = Pt(6)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        for run in para.runs:
            run.font.size = Pt(14)
            run.font.name = "Times New Roman"


def extract_formatting_from_ai(instructions=None, reference_image=None):
    """
    Use Gemini AI to understand formatting requirements from text or image.
    Returns a JSON-like dict with formatting specifications.
    Enhanced version with better prompt engineering for accurate formatting extraction.
    """
    import json
    
    default_format = {
        "font_name": "Times New Roman",
        "font_name_arabic": "Simplified Arabic",
        "font_size": 14,
        "line_spacing": 1.5,
        "alignment": "justify",
        "margins": {"top": 2.5, "bottom": 2.5, "left": 2.5, "right": 2.5},
        "paragraph_spacing_before": 0,
        "paragraph_spacing_after": 6,
        "first_line_indent": 1.27,
        "heading_font_size": 16,
        "heading_bold": True,
        "heading_alignment": "center",
        "rtl_direction": True
    }
    
    result = None
    
    prompt = """أنت خبير متخصص في تنسيق المستندات الأكاديمية والرسمية. مهمتك تحليل متطلبات التنسيق وإرجاع مواصفات دقيقة بصيغة JSON.

قم بإرجاع JSON فقط بالضبط بهذا الشكل (بدون أي نص إضافي قبله أو بعده):
{
    "font_name": "Times New Roman",
    "font_name_arabic": "Simplified Arabic",
    "font_size": 14,
    "line_spacing": 1.5,
    "alignment": "justify",
    "margins": {"top": 2.5, "bottom": 2.5, "left": 2.5, "right": 2.5},
    "paragraph_spacing_before": 0,
    "paragraph_spacing_after": 6,
    "first_line_indent": 1.27,
    "heading_font_size": 16,
    "heading_bold": true,
    "heading_alignment": "center",
    "rtl_direction": true
}

ملاحظات مهمة:
- font_name: الخط الإنجليزي (Times New Roman, Arial, Calibri)
- font_name_arabic: الخط العربي (Simplified Arabic, Traditional Arabic, Arial)
- font_size: حجم الخط بالنقاط (12, 14, 16)
- line_spacing: تباعد الأسطر (1.0, 1.15, 1.5, 2.0)
- alignment: محاذاة النص (justify, right, left, center)
- margins: الهوامش بالسنتيمتر
- first_line_indent: المسافة البادئة للسطر الأول بالسنتيمتر
- heading_alignment: محاذاة العناوين (center, right, left)
- rtl_direction: اتجاه النص من اليمين لليسار للعربية

"""
    
    if instructions:
        prompt += f"\n\nوصف التنسيق المطلوب:\n{instructions}\n\nقم بتحليل هذا الوصف وأرجع JSON فقط بالمواصفات المناسبة."
    else:
        prompt += "\n\nلم يتم تقديم تعليمات محددة. أرجع التنسيق الافتراضي للمستندات الأكاديمية العربية."
    
    try:
        if reference_image:
            prompt += "\n\nقم أيضاً بتحليل الصورة المرفقة واستخرج أسلوب التنسيق منها."
            result = call_gemini_vision(reference_image, prompt)
        else:
            result = call_gemini_text(prompt)
        
        if not result:
            logging.warning("AI returned no result, using default format")
            return default_format
        
        result_clean = result.strip()
        
        if "```json" in result_clean:
            start = result_clean.find("```json") + 7
            end = result_clean.find("```", start)
            if end > start:
                result_clean = result_clean[start:end].strip()
        elif "```" in result_clean:
            start = result_clean.find("```") + 3
            end = result_clean.find("```", start)
            if end > start:
                result_clean = result_clean[start:end].strip()
        
        if result_clean.startswith("{"):
            brace_count = 0
            json_end = 0
            for i, char in enumerate(result_clean):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
            if json_end > 0:
                result_clean = result_clean[:json_end]
        
        format_spec = json.loads(result_clean)
        
        for key in default_format:
            if key not in format_spec:
                format_spec[key] = default_format[key]
        
        logging.info(f"Successfully extracted format spec: {format_spec}")
        return format_spec
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse AI format response: {e}")
        logging.error(f"Raw response: {result[:500] if result else 'None'}")
        return default_format
    except Exception as e:
        logging.error(f"Error in extract_formatting_from_ai: {e}")
        return default_format


def apply_ai_format(doc: Document, format_spec: dict):
    """
    Apply AI-generated formatting specifications to a Word document.
    Enhanced version that properly applies all formatting to every paragraph and run.
    Supports documents up to 100+ pages.
    """
    from docx.shared import RGBColor, Twips
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    
    alignment_map = {
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER
    }
    
    margins = format_spec.get("margins", {})
    for section in doc.sections:
        try:
            section.top_margin = Cm(float(margins.get("top", 2.5)))
            section.bottom_margin = Cm(float(margins.get("bottom", 2.5)))
            section.left_margin = Cm(float(margins.get("left", 2.5)))
            section.right_margin = Cm(float(margins.get("right", 2.5)))
        except Exception as e:
            logging.warning(f"Could not set margins: {e}")
    
    font_name = format_spec.get("font_name", "Times New Roman")
    font_name_arabic = format_spec.get("font_name_arabic", "Simplified Arabic")
    font_size = float(format_spec.get("font_size", 14))
    line_spacing = float(format_spec.get("line_spacing", 1.5))
    alignment_str = format_spec.get("alignment", "justify")
    alignment = alignment_map.get(alignment_str, WD_ALIGN_PARAGRAPH.JUSTIFY)
    para_spacing_before = float(format_spec.get("paragraph_spacing_before", 0))
    para_spacing_after = float(format_spec.get("paragraph_spacing_after", 6))
    first_indent = float(format_spec.get("first_line_indent", 0))
    heading_size = float(format_spec.get("heading_font_size", 16))
    heading_bold = format_spec.get("heading_bold", True)
    heading_alignment_str = format_spec.get("heading_alignment", "center")
    heading_alignment = alignment_map.get(heading_alignment_str, WD_ALIGN_PARAGRAPH.CENTER)
    rtl_direction = format_spec.get("rtl_direction", True)
    
    try:
        for style_name in ["Normal", "Body Text", "Default Paragraph Font"]:
            try:
                style = doc.styles[style_name]
                style.font.name = font_name
                style.font.size = Pt(font_size)
                style._element.rPr.rFonts.set(qn('w:eastAsia'), font_name_arabic)
            except:
                pass
    except Exception as e:
        logging.warning(f"Could not set document styles: {e}")
    
    def is_arabic(text):
        """Check if text contains Arabic characters"""
        arabic_range = range(0x0600, 0x06FF + 1)
        return any(ord(char) in arabic_range for char in text)
    
    def is_heading(para):
        """Determine if a paragraph is a heading"""
        text = para.text.strip()
        if not text:
            return False
        if para.style and para.style.name and para.style.name.startswith("Heading"):
            return True
        if len(text) < 80 and not text.endswith(('.', '،', '؟', '!', ':', ';')):
            if para.runs and any(run.bold for run in para.runs if run.text.strip()):
                return True
        if len(text) < 50 and text.isupper():
            return True
        return False
    
    total_paragraphs = len(doc.paragraphs)
    formatted_count = 0
    
    for idx, para in enumerate(doc.paragraphs):
        try:
            text = para.text.strip()
            if not text:
                continue
            
            formatted_count += 1
            is_head = is_heading(para)
            text_is_arabic = is_arabic(text)
            
            pf = para.paragraph_format
            pf.line_spacing = line_spacing
            pf.space_before = Pt(para_spacing_before)
            pf.space_after = Pt(para_spacing_after)
            
            if is_head:
                pf.alignment = heading_alignment
                pf.first_line_indent = Cm(0)
            else:
                pf.alignment = alignment
                if first_indent > 0:
                    pf.first_line_indent = Cm(first_indent)
            
            if rtl_direction and text_is_arabic:
                try:
                    pPr = para._element.get_or_add_pPr()
                    bidi = OxmlElement('w:bidi')
                    bidi.set(qn('w:val'), '1')
                    pPr.append(bidi)
                except:
                    pass
            
            current_font = font_name_arabic if text_is_arabic else font_name
            current_size = heading_size if is_head else font_size
            
            if para.runs:
                for run in para.runs:
                    try:
                        run.font.name = current_font
                        run.font.size = Pt(current_size)
                        
                        run._element.rPr.rFonts.set(qn('w:ascii'), font_name)
                        run._element.rPr.rFonts.set(qn('w:hAnsi'), font_name)
                        run._element.rPr.rFonts.set(qn('w:cs'), font_name_arabic)
                        run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name_arabic)
                        
                        if is_head:
                            run.font.bold = heading_bold
                    except Exception as run_error:
                        logging.debug(f"Could not format run: {run_error}")
            else:
                try:
                    run = para.add_run(text)
                    para.clear()
                    para.add_run(text)
                    for run in para.runs:
                        run.font.name = current_font
                        run.font.size = Pt(current_size)
                        if is_head:
                            run.font.bold = heading_bold
                except:
                    pass
                    
        except Exception as para_error:
            logging.warning(f"Error formatting paragraph {idx}: {para_error}")
            continue
    
    for table in doc.tables:
        try:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        text = para.text.strip()
                        if not text:
                            continue
                        text_is_arabic = is_arabic(text)
                        current_font = font_name_arabic if text_is_arabic else font_name
                        
                        for run in para.runs:
                            try:
                                run.font.name = current_font
                                run.font.size = Pt(font_size)
                            except:
                                pass
        except Exception as table_error:
            logging.warning(f"Error formatting table: {table_error}")
    
    logging.info(f"Successfully formatted {formatted_count}/{total_paragraphs} paragraphs")


def convert_pdf_to_docx_for_formatting(pdf_path, output_path):
    """Convert PDF to DOCX for formatting purposes."""
    try:
        from pdf2docx import Converter
        cv = Converter(pdf_path)
        cv.convert(output_path, start=0, end=None)
        cv.close()
        return True
    except Exception as e:
        logging.error(f"PDF to DOCX conversion failed: {e}")
        return False


@app.route("/format-docx", methods=["POST"])
def format_docx():
    uploaded = request.files.get("file")

    if not uploaded or uploaded.filename == "":
        return jsonify({'error': 'من فضلك ارفع ملف DOCX'}), 400

    filename = uploaded.filename.lower()
    if not filename.endswith(".docx"):
        return jsonify({'error': 'النسخة الحالية تدعم ملفات Word بصيغة .docx فقط'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input.docx")
        out_path = os.path.join(tmpdir, "formatted.docx")

        uploaded.save(in_path)

        doc = Document(in_path)
        apply_basic_format(doc)
        doc.save(out_path)

        download_name = f"formatted_{uploaded.filename}"
        return send_file(out_path, as_attachment=True, download_name=download_name)


@app.route("/format-document-ai", methods=["POST"])
def format_document_ai():
    """
    AI-powered document formatting endpoint.
    Accepts: main document (DOCX/PDF), optional formatting instructions, optional reference file (image/DOCX/PDF)
    """
    uploaded = request.files.get("file")
    reference_file = request.files.get("reference_file")
    instructions = request.form.get("instructions", "").strip()
    
    if not uploaded or uploaded.filename == "":
        return jsonify({'error': 'من فضلك ارفع ملف للتنسيق'}), 400
    
    filename = uploaded.filename.lower()
    if not filename.endswith((".docx", ".pdf")):
        return jsonify({'error': 'يدعم ملفات Word (.docx) و PDF فقط'}), 400
    
    if not instructions and not reference_file:
        return jsonify({'error': 'يرجى إدخال وصف التنسيق أو رفع ملف مرجعي'}), 400
    
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "input" + os.path.splitext(uploaded.filename)[1])
        docx_path = os.path.join(tmpdir, "document.docx")
        out_path = os.path.join(tmpdir, "formatted.docx")
        
        uploaded.save(in_path)
        
        if filename.endswith(".pdf"):
            logging.info("Converting PDF to DOCX for formatting...")
            if not convert_pdf_to_docx_for_formatting(in_path, docx_path):
                return jsonify({'error': 'فشل في تحويل ملف PDF. جرب ملف Word مباشرة.'}), 500
        else:
            docx_path = in_path
        
        reference_image = None
        if reference_file and reference_file.filename:
            ref_filename = reference_file.filename.lower()
            ref_path = os.path.join(tmpdir, "reference" + os.path.splitext(reference_file.filename)[1])
            reference_file.save(ref_path)
            
            if ref_filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                reference_image = ref_path
            elif ref_filename.endswith(".pdf"):
                try:
                    import fitz
                    pdf_doc = fitz.open(ref_path)
                    page = pdf_doc[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_path = os.path.join(tmpdir, "ref_screenshot.png")
                    pix.save(img_path)
                    pdf_doc.close()
                    reference_image = img_path
                    logging.info("Successfully extracted image from reference PDF")
                except Exception as e:
                    logging.error(f"Failed to extract image from reference PDF: {e}")
                    instructions += "\n\nالملف المرجعي هو PDF. استخدم تنسيق بحثي أكاديمي مشابه."
            elif ref_filename.endswith(".docx"):
                try:
                    ref_doc = Document(ref_path)
                    ref_format_info = []
                    
                    if ref_doc.sections:
                        section = ref_doc.sections[0]
                        margins = {
                            "top": round(section.top_margin.cm, 1) if section.top_margin else 2.5,
                            "bottom": round(section.bottom_margin.cm, 1) if section.bottom_margin else 2.5,
                            "left": round(section.left_margin.cm, 1) if section.left_margin else 2.5,
                            "right": round(section.right_margin.cm, 1) if section.right_margin else 2.5
                        }
                        ref_format_info.append(f"الهوامش: أعلى {margins['top']}سم، أسفل {margins['bottom']}سم، يسار {margins['left']}سم، يمين {margins['right']}سم")
                    
                    for para in ref_doc.paragraphs[:5]:
                        if para.runs:
                            run = para.runs[0]
                            if run.font.name:
                                ref_format_info.append(f"الخط: {run.font.name}")
                            if run.font.size:
                                ref_format_info.append(f"حجم الخط: {run.font.size.pt} نقطة")
                            break
                    
                    for para in ref_doc.paragraphs[:5]:
                        if para.paragraph_format.line_spacing:
                            ref_format_info.append(f"تباعد الأسطر: {para.paragraph_format.line_spacing}")
                            break
                    
                    if ref_format_info:
                        instructions += "\n\nالتنسيق المستخرج من الملف المرجعي:\n" + "\n".join(ref_format_info)
                    else:
                        instructions += "\n\nالملف المرجعي هو مستند Word. استخدم تنسيق بحثي أكاديمي مشابه."
                    
                    logging.info(f"Extracted formatting from reference DOCX: {ref_format_info}")
                except Exception as e:
                    logging.error(f"Failed to extract formatting from reference DOCX: {e}")
                    instructions += "\n\nالملف المرجعي هو مستند Word. استخدم تنسيق بحثي أكاديمي مشابه."
        
        logging.info(f"Extracting formatting specs from AI... Instructions: {instructions[:100] if instructions else 'None'}, Reference: {bool(reference_image)}")
        format_spec = extract_formatting_from_ai(instructions, reference_image)
        logging.info(f"AI format spec: {format_spec}")
        
        doc = Document(docx_path)
        apply_ai_format(doc, format_spec)
        doc.save(out_path)
        
        original_name = os.path.splitext(uploaded.filename)[0]
        download_name = f"formatted_{original_name}.docx"
        
        return send_file(out_path, as_attachment=True, download_name=download_name)


# Extract audio from YouTube for background
@app.route('/get-audio', methods=['POST'])
def get_audio():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        temp_dir = tempfile.gettempdir()
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '128',
            }],
            'outtmpl': os.path.join(temp_dir, 'arkan_audio_%(id)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_file = os.path.join(temp_dir, f"arkan_audio_{info['id']}.mp3")
            
            if os.path.exists(audio_file):
                @after_this_request
                def cleanup(response):
                    safe_remove_file(audio_file)
                    return response
                
                return send_file(audio_file, mimetype='audio/mpeg', as_attachment=False, download_name=f"audio_{info['id']}.mp3")
            else:
                return jsonify({'error': 'Failed to extract audio'}), 500
    
    except Exception as e:
        logging.error(f"Audio extraction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ADMIN DASHBOARD ROUTES
# =============================================================================
# Protected admin dashboard for Gemini API key management
# Access only via: /admin/keys (requires admin login)
# =============================================================================

ADMIN_DASHBOARD_PATH = '/admin/keys'


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page."""
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = AdminUser.query.filter_by(username=username).first()
        if user and user.check_password(password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user)
            logging.info(f"[Admin] User '{username}' logged in successfully")
            return redirect(url_for('admin_dashboard'))
        
        flash('اسم المستخدم أو كلمة المرور غير صحيحة', 'error')
    
    return render_template('admin_login.html')


@app.route('/admin/logout')
@login_required
def admin_logout():
    """Admin logout."""
    logout_user()
    return redirect(url_for('admin_login'))


@app.route(ADMIN_DASHBOARD_PATH)
@login_required
def admin_dashboard():
    """Admin dashboard for Gemini API key management."""
    if not current_user.is_admin:
        flash('ليس لديك صلاحية الوصول إلى لوحة التحكم', 'error')
        return redirect(url_for('admin_login'))
    
    gemini_manager.init_db_state()
    
    stats = gemini_manager.get_dashboard_stats()
    keys_info = gemini_manager.get_all_keys_info()
    hourly_stats = gemini_manager.get_hourly_stats()
    
    today = date.today()
    daily_db_stats = DailyStats.query.filter_by(date=today).first()
    
    hourly_data = []
    for hour in range(24):
        hour_stats = hourly_stats.get(hour, {})
        hourly_data.append({
            'hour': hour,
            'total': hour_stats.get('total', 0),
            'success': hour_stats.get('success', 0),
            'quota_errors': hour_stats.get('quota_errors', 0)
        })
    
    show_warning = stats['quota_errors_today'] > 10
    
    return render_template('admin_dashboard.html',
                           stats=stats,
                           keys_info=keys_info,
                           hourly_data=hourly_data,
                           daily_stats=daily_db_stats,
                           show_warning=show_warning)


@app.route('/admin/api/toggle-key', methods=['POST'])
@login_required
def admin_toggle_key():
    """API endpoint to toggle key enabled/disabled state."""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    key_index = data.get('key_index')
    disabled = data.get('disabled', True)
    reason = data.get('reason', 'تعطيل يدوي من لوحة التحكم')
    
    if key_index is None or key_index < 0 or key_index >= len(gemini_manager.keys):
        return jsonify({'error': 'Invalid key index'}), 400
    
    gemini_manager.set_key_disabled(key_index, disabled, reason)
    
    return jsonify({
        'success': True,
        'key_index': key_index,
        'disabled': disabled,
        'status': gemini_manager.get_key_status(key_index)
    })


@app.route('/admin/api/stats')
@login_required
def admin_api_stats():
    """API endpoint to get current stats (for dashboard refresh)."""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    return jsonify({
        'stats': gemini_manager.get_dashboard_stats(),
        'keys_info': gemini_manager.get_all_keys_info(),
        'hourly_stats': gemini_manager.get_hourly_stats()
    })


def init_admin_user():
    """Initialize default admin user if none exists."""
    try:
        admin = AdminUser.query.filter_by(is_admin=True).first()
        if not admin:
            admin = AdminUser(username='admin', is_admin=True)
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            logging.info("[Admin] Default admin user created (username: admin, password: admin123)")
            logging.info("[Admin] IMPORTANT: Please change the default password immediately!")
    except Exception as e:
        logging.warning(f"[Admin] Could not initialize admin user: {e}")


with app.app_context():
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    existing_tables = inspector.get_table_names()
    if not existing_tables:
        db.create_all()
        logging.info("[DB] Database tables created successfully")
    else:
        logging.info(f"[DB] Database already initialized with {len(existing_tables)} tables")
    init_admin_user()
    gemini_manager.init_db_state()
    logging.info("=" * 60)
    logging.info(f"Admin dashboard for Gemini keys is available at: {ADMIN_DASHBOARD_PATH} (requires admin login)")
    logging.info("Default admin credentials: username=admin, password=admin123")
    logging.info("=" * 60)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
