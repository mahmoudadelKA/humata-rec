import os
import time
import logging
import requests
import tempfile

logger = logging.getLogger(__name__)

REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"

def separate_vocals_with_replicate(audio_file_path: str, max_wait_seconds: int = 300) -> str:
    """
    Use Replicate's Demucs model to separate vocals from music.
    
    Args:
        audio_file_path: Path to the input audio file
        max_wait_seconds: Maximum time to wait for processing (default 5 minutes)
    
    Returns:
        Path to the downloaded vocals-only audio file
    
    Raises:
        ValueError: If API token is not configured
        Exception: If separation fails
    """
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN غير مُعد. يرجى إضافة مفتاح API من Replicate.com")
    
    headers = {
        "Authorization": f"Token {api_token}",
        "Content-Type": "application/json"
    }
    
    logger.info(f"Uploading audio file to Replicate: {audio_file_path}")
    upload_url = upload_file_to_replicate(audio_file_path, api_token)
    
    payload = {
        "version": "25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953",
        "input": {
            "audio": upload_url,
            "stem": "vocals",
            "model_name": "htdemucs",
            "shifts": 1,
            "overlap": 0.25,
            "clip_mode": "rescale",
            "mp3_bitrate": 192,
            "output_format": "mp3",
            "float32": False
        }
    }
    
    logger.info("Starting Demucs separation job on Replicate...")
    response = requests.post(REPLICATE_API_URL, json=payload, headers=headers, timeout=30)
    
    if response.status_code != 201:
        error_msg = response.json().get("detail", response.text)
        logger.error(f"Replicate API error: {error_msg}")
        raise Exception(f"فشل بدء عملية الفصل: {error_msg}")
    
    prediction = response.json()
    prediction_id = prediction["id"]
    get_url = prediction["urls"]["get"]
    
    logger.info(f"Prediction started with ID: {prediction_id}")
    
    start_time = time.time()
    poll_interval = 2
    
    while time.time() - start_time < max_wait_seconds:
        time.sleep(poll_interval)
        
        status_response = requests.get(get_url, headers=headers, timeout=30)
        status_data = status_response.json()
        status = status_data.get("status")
        
        logger.info(f"Prediction status: {status}")
        
        if status == "succeeded":
            output = status_data.get("output")
            logger.info(f"Replicate output received: {output}")
            if output:
                vocals_url = None
                if isinstance(output, str):
                    vocals_url = output
                elif isinstance(output, dict):
                    vocals_url = output.get("vocals") or output.get("audio") or output.get("output")
                    if not vocals_url:
                        for key in output:
                            if output[key] and isinstance(output[key], str) and output[key].startswith("http"):
                                vocals_url = output[key]
                                break
                elif isinstance(output, list) and len(output) > 0:
                    vocals_url = output[0]
                
                if vocals_url:
                    logger.info(f"Separation complete. Downloading vocals from: {vocals_url}")
                    return download_vocals_file(vocals_url)
                else:
                    logger.error(f"Could not extract vocals URL from output: {output}")
                    raise Exception("لم يتم العثور على رابط ملف الصوت المفصول")
            else:
                raise Exception("لم يتم إرجاع ملف الصوت من الخدمة")
        
        elif status == "failed":
            error = status_data.get("error", "Unknown error")
            logger.error(f"Replicate prediction failed: {error}")
            raise Exception(f"فشل فصل الصوت: {error}")
        
        elif status == "canceled":
            raise Exception("تم إلغاء عملية الفصل")
        
        poll_interval = min(poll_interval * 1.5, 10)
    
    raise Exception(f"انتهت مهلة الانتظار ({max_wait_seconds} ثانية). حاول مع ملف أقصر.")


def upload_file_to_replicate(file_path: str, api_token: str) -> str:
    """
    Upload a file to Replicate's file hosting and return the URL.
    
    Args:
        file_path: Path to the file to upload
        api_token: Replicate API token
    
    Returns:
        URL of the uploaded file
    """
    headers = {
        "Authorization": f"Token {api_token}"
    }
    
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    
    upload_request_url = "https://api.replicate.com/v1/files"
    
    with open(file_path, "rb") as f:
        files = {"file": (file_name, f)}
        response = requests.post(upload_request_url, headers=headers, files=files, timeout=300)
    
    if response.status_code in [200, 201]:
        file_data = response.json()
        file_url = file_data.get("urls", {}).get("get", file_data.get("url"))
        if file_url:
            logger.info(f"File uploaded successfully: {file_url}")
            return file_url
    
    logger.info("Using direct file upload as data URL...")
    import base64
    with open(file_path, "rb") as f:
        audio_data = f.read()
    
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".webm": "audio/webm"
    }
    mime_type = mime_types.get(ext, "audio/mpeg")
    
    data_url = f"data:{mime_type};base64,{base64.b64encode(audio_data).decode('utf-8')}"
    return data_url


def download_vocals_file(url: str) -> str:
    """
    Download the vocals file from Replicate output URL.
    
    Args:
        url: URL to download the vocals file from
    
    Returns:
        Path to the downloaded file
    """
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"vocals_{int(time.time())}.wav")
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Vocals file downloaded to: {output_path}")
    return output_path
