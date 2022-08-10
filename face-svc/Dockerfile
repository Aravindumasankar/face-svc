FROM python:3.6
LABEL maintainer="aravind-umasankar"
RUN apt-get update && apt-get install -y cmake
WORKDIR /app
COPY . /app/
RUN pip3 install -r face-svc/requirements.txt
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["face-svc/app/app.py"]