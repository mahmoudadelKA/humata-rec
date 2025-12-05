# 1. استخدام صورة بايثون الرسمية الخفيفة
FROM python:3.10-slim

# 2. إعداد متغيرات البيئة
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. تثبيت مكتبات النظام الضرورية (FFmpeg + Tesseract + دعم العربية)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-ara \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. تحديد مجلد العمل
WORKDIR /app

# 5. نسخ الملفات وتثبيت المكتبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. نسخ باقي المشروع
COPY . .

# 7. فتح المنفذ
EXPOSE 8080

# 8. أمر التشغيل (مع مهلة ساعة)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "3600", "--workers", "2", "--threads", "4", "main:app"]
