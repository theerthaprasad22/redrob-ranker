# Redrob ranker — sandbox image.
#   docker build -t redrob-ranker .
#   docker run -p 8501:8501 redrob-ranker          # Streamlit sandbox UI
#   docker run redrob-ranker python rank.py --candidates /data/candidates.jsonl --out /data/submission.csv
FROM python:3.11-slim

WORKDIR /app

# Core deps + streamlit for the sandbox UI. (sentence-transformers is optional
# and intentionally NOT installed here to keep the image small and the ranking
# step dependency-light.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit

COPY . .

EXPOSE 8501

# Default: launch the sandbox UI. Override the CMD to run rank.py instead.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
