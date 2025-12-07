from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


class AdminUser(UserMixin, db.Model):
    __tablename__ = 'admin_users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class AIProviderState(db.Model):
    """Track state of AI providers (Groq, HuggingFace)"""
    __tablename__ = 'ai_provider_states'
    
    id = db.Column(db.Integer, primary_key=True)
    provider_name = db.Column(db.String(50), unique=True, nullable=False)
    is_enabled = db.Column(db.Boolean, default=True)
    is_healthy = db.Column(db.Boolean, default=True)
    last_health_check = db.Column(db.DateTime)
    total_requests = db.Column(db.Integer, default=0)
    successful_requests = db.Column(db.Integer, default=0)
    failed_requests = db.Column(db.Integer, default=0)
    last_error = db.Column(db.String(500))
    last_error_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AIUsageLog(db.Model):
    """Log AI API requests for analytics"""
    __tablename__ = 'ai_usage_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    provider = db.Column(db.String(50), nullable=False)
    operation_type = db.Column(db.String(50), nullable=False)
    success = db.Column(db.Boolean, nullable=False)
    error_type = db.Column(db.String(100))
    error_message = db.Column(db.String(500))
    duration_seconds = db.Column(db.Float)
    session_id = db.Column(db.String(100))
    cached = db.Column(db.Boolean, default=False)
    hour_bucket = db.Column(db.Integer)


class DailyStats(db.Model):
    """Daily aggregated statistics"""
    __tablename__ = 'daily_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, unique=True, nullable=False, index=True)
    total_requests = db.Column(db.Integer, default=0)
    successful_requests = db.Column(db.Integer, default=0)
    failed_requests = db.Column(db.Integer, default=0)
    cached_requests = db.Column(db.Integer, default=0)
    llm_requests = db.Column(db.Integer, default=0)
    whisper_requests = db.Column(db.Integer, default=0)
    vision_requests = db.Column(db.Integer, default=0)
    groq_requests = db.Column(db.Integer, default=0)
    huggingface_requests = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RequestCache(db.Model):
    """Persistent cache for AI requests"""
    __tablename__ = 'request_cache'
    
    id = db.Column(db.Integer, primary_key=True)
    cache_key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    operation_type = db.Column(db.String(50), nullable=False)
    result = db.Column(db.Text, nullable=False)
    file_hash = db.Column(db.String(64))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime, default=datetime.utcnow)
    access_count = db.Column(db.Integer, default=1)


class ActiveSession(db.Model):
    """Track active user sessions"""
    __tablename__ = 'active_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    user_id = db.Column(db.String(100))
    ip_address = db.Column(db.String(45), nullable=False)
    user_agent = db.Column(db.String(500))
    device_type = db.Column(db.String(50))
    browser = db.Column(db.String(100))
    os_name = db.Column(db.String(100))
    country = db.Column(db.String(100))
    first_seen = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    page_views = db.Column(db.Integer, default=1)
    ai_requests = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Boolean, default=True)


class SessionRateLimit(db.Model):
    """Track rate limits per session"""
    __tablename__ = 'session_rate_limits'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    request_count = db.Column(db.Integer, default=0)
    audio_count = db.Column(db.Integer, default=0)
    long_audio_count = db.Column(db.Integer, default=0)
    window_start = db.Column(db.DateTime, default=datetime.utcnow)
    last_request = db.Column(db.DateTime, default=datetime.utcnow)
    is_blocked = db.Column(db.Boolean, default=False)
    blocked_until = db.Column(db.DateTime)
    block_reason = db.Column(db.String(200))
