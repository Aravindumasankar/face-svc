FROM python:3.6
LABEL maintainer="aravind-umasankar"
RUN apt-get update && apt-get install -y cmake
RUN apt-get install ffmpeg libsm6 libxext6 libgl1 python3-opencv wget -y
WORKDIR /app
COPY . /app/
RUN wget https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights -P /app/object-svc/app/object_detection/yolo-coco
RUN pip3 install -r object-svc/requirements.txt
WORKDIR /app/object-svc/app/
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["app.py"]