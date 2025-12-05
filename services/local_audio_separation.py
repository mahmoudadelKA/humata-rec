import os
import subprocess
import tempfile
import logging
import shutil

logger = logging.getLogger(__name__)

def separate_vocals_local(audio_file_path: str) -> str:
    """
    Use FFmpeg audio filters to isolate vocals from music (free, local processing).
    
    This approach uses multiple audio filters:
    1. Center channel extraction (vocals are usually centered)
    2. High-pass filter to remove bass frequencies (often instrumental)
    3. Dynamic compression to enhance vocal clarity
    
    Args:
        audio_file_path: Path to the input audio file
    
    Returns:
        Path to the processed vocals-only audio file
    
    Note: This is a simplified approach that works well for most pop/spoken content.
          For professional-grade separation, dedicated AI models are needed.
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"ملف الصوت غير موجود: {audio_file_path}")
    
    temp_dir = tempfile.gettempdir()
    output_filename = f"vocals_{os.path.basename(audio_file_path).rsplit('.', 1)[0]}_{int(os.path.getmtime(audio_file_path))}.mp3"
    output_path = os.path.join(temp_dir, output_filename)
    
    logger.info(f"Starting local vocal isolation: {audio_file_path}")
    
    vocal_filter = (
        "pan=stereo|c0=c0-c1|c1=c1-c0,"
        "highpass=f=100,"
        "lowpass=f=8000,"
        "acompressor=threshold=-20dB:ratio=4:attack=5:release=50,"
        "dynaudnorm=p=0.9:m=100:s=12,"
        "volume=1.5"
    )
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', audio_file_path,
        '-af', vocal_filter,
        '-acodec', 'libmp3lame',
        '-b:a', '192k',
        '-ar', '44100',
        output_path
    ]
    
    try:
        logger.info(f"Running FFmpeg vocal isolation...")
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            logger.warning(f"FFmpeg center isolation warning: {result.stderr[:500]}")
            
            simple_filter = "highpass=f=200,lowpass=f=6000,volume=1.3"
            ffmpeg_simple = [
                'ffmpeg', '-y',
                '-i', audio_file_path,
                '-af', simple_filter,
                '-acodec', 'libmp3lame',
                '-b:a', '192k',
                output_path
            ]
            
            result = subprocess.run(ffmpeg_simple, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr[:300]}")
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Vocal isolation complete: {output_path}")
            return output_path
        else:
            raise Exception("لم يتم إنشاء ملف الصوت الناتج")
            
    except subprocess.TimeoutExpired:
        raise Exception("انتهت مهلة المعالجة. جرب ملفاً أقصر.")
    except Exception as e:
        logger.error(f"Local vocal isolation error: {e}")
        raise


def separate_vocals_enhanced(audio_file_path: str) -> str:
    """
    Enhanced vocal isolation using multiple FFmpeg techniques.
    Combines center channel extraction with frequency filtering.
    
    Args:
        audio_file_path: Path to the input audio file
    
    Returns:
        Path to the processed vocals-only audio file
    """
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"ملف الصوت غير موجود: {audio_file_path}")
    
    temp_dir = tempfile.gettempdir()
    intermediate_path = os.path.join(temp_dir, f"intermediate_{int(os.path.getmtime(audio_file_path))}.wav")
    output_path = os.path.join(temp_dir, f"vocals_enhanced_{int(os.path.getmtime(audio_file_path))}.mp3")
    
    try:
        logger.info("Step 1: Center channel extraction...")
        center_cmd = [
            'ffmpeg', '-y',
            '-i', audio_file_path,
            '-af', 'pan=stereo|c0=c0-c1|c1=c1-c0',
            '-ar', '44100',
            intermediate_path
        ]
        
        result = subprocess.run(center_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0 or not os.path.exists(intermediate_path):
            logger.warning("Center extraction failed, using original file")
            intermediate_path = audio_file_path
        
        logger.info("Step 2: Applying vocal enhancement filters...")
        enhance_filter = (
            "highpass=f=120,"
            "lowpass=f=7500,"
            "equalizer=f=250:t=q:w=2:g=3,"
            "equalizer=f=2000:t=q:w=1:g=2,"
            "equalizer=f=4000:t=q:w=1:g=1,"
            "acompressor=threshold=-25dB:ratio=3:attack=10:release=100,"
            "dynaudnorm=p=0.95:m=50:s=8,"
            "volume=1.4"
        )
        
        enhance_cmd = [
            'ffmpeg', '-y',
            '-i', intermediate_path,
            '-af', enhance_filter,
            '-acodec', 'libmp3lame',
            '-b:a', '192k',
            output_path
        ]
        
        result = subprocess.run(enhance_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.warning("Enhanced processing failed, falling back to simple method")
            return separate_vocals_local(audio_file_path)
        
        if intermediate_path != audio_file_path and os.path.exists(intermediate_path):
            try:
                os.remove(intermediate_path)
            except:
                pass
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Enhanced vocal isolation complete: {output_path}")
            return output_path
        else:
            raise Exception("فشل إنشاء ملف الصوت")
            
    except Exception as e:
        if intermediate_path != audio_file_path and os.path.exists(intermediate_path):
            try:
                os.remove(intermediate_path)
            except:
                pass
        logger.error(f"Enhanced vocal isolation error: {e}")
        raise
