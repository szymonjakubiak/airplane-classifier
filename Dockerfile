FROM python:3.8
WORKDIR /dash-app
COPY requirements.txt .
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
RUN gdown https://drive.google.com/uc?id=1t4xO8wEe9H2E7eNnjaiY1_VRrjzKNVwX
COPY dash_app.py .
EXPOSE 8000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "dash_app:server"]
