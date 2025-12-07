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


class GeminiKeyState(db.Model):
    __tablename__ = 'gemini_key_states'
    
    id = db.Column(db.Integer, primary_key=True)
    key_index = db.Column(db.Integer, unique=True, nullable=False)
    key_name = db.Column(db.String(50), nullable=False)
    is_manually_disabled = db.Column(db.Boolean, default=False)
    disabled_at = db.Column(db.DateTime)
    disabled_reason = db.Column(db.String(200))
    total_requests = db.Column(db.Integer, default=0)
    successful_requests = db.Column(db.Integer, default=0)
    failed_requests = db.Column(db.Integer, default=0)
    cooldown_count = db.Column(db.Integer, default=0)
    last_used_at = db.Column(db.DateTime)
    last_cooldown_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GeminiUsageLog(db.Model):
    __tablename__ = 'gemini_usage_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    key_index = db.Column(db.Integer, nullable=False)
    operation_type = db.Column(db.String(50), nullable=False)
    is_retry = db.Column(db.Boolean, default=False)
    success = db.Column(db.Boolean, nullable=False)
    error_type = db.Column(db.String(50))
    content_duration_minutes = db.Column(db.Float)
    session_id = db.Column(db.String(100))
    hour_bucket = db.Column(db.Integer)


class DailyStats(db.Model):
    __tablename__ = 'daily_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, unique=True, nullable=False, index=True)
    total_requests = db.Column(db.Integer, default=0)
    successful_requests = db.Column(db.Integer, default=0)
    failed_requests = db.Column(db.Integer, default=0)
    quota_errors = db.Column(db.Integer, default=0)
    text_requests = db.Column(db.Integer, default=0)
    vision_requests = db.Column(db.Integer, default=0)
    audio_requests = db.Column(db.Integer, default=0)
    video_requests = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
