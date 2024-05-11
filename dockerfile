FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/DataPipeline

RUN python download_model.py

WORKDIR /app

EXPOSE 80

CMD ["python", "-m", "streamlit", "run", "UI.py"]
