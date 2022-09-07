FROM python:3.6
LABEL maintainer="aravind-umasankar"
RUN apt-get update && apt-get install -y cmake
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 python3-opencv -y
COPY . /app/
WORKDIR /app/people_counter-svc 
RUN pip3 install -r requirements.txt
EXPOSE 8000
ENTRYPOINT ["python"]
CMD ["app.py"]