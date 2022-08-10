FROM python:3.6
LABEL maintainer="aravind-umasankar"
RUN apt-get update && apt-get install -y cmake
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 python3-opencv -y
WORKDIR /app
COPY . /app/
RUN pip3 install -r face-svc/requirements.txt
WORKDIR /app/face-svc/app/
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["app.py"]