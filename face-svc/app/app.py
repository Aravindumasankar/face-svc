import os
import uuid
import face.recognition as face_recognition
import face.emotion as emotion
from flask import Flask, request, abort, jsonify

app = Flask(__name__)


@app.route('/')
def info():
    data = {
        "Name": "Face-svc",
        "version": "v1",
        "Author": "Aravind Umasankar",
        "Description": "Face Utility Service"
    }
    return jsonify(data)


allowed_ext = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


@app.route('/classify', methods=['POST'])
def upload_file():
    data = {}
    upload_dir = 'uploads/'
    if not os.path.exists(upload_dir):
        print("Uploads directory created")
        os.makedirs(upload_dir)
    if 'file' not in request.files:
        data['msg'] = 'No file part in the request'
        resp = jsonify(data)
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        data['msg'] = 'No file selected for uploading'
        resp = jsonify(data)
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        file_name, file_extension = os.path.splitext(file.filename)
        data['original_file'] = file.filename
        data['_id'] = str(uuid.uuid4())
        file_name = data['_id'] + file_extension
        file.save(os.path.join(upload_dir, file_name))
        data['file_name'] = file_name
        data['file_extension'] = file_extension
        data['file_path'] = str(upload_dir + data['file_name'])
        today_model_file = 'face/recognition/model/vadivelu_trained_knn_model_.clf'
        data['face_recogniton'] = face_recognition.predict(request.url_root, data['file_name'], data['file_path'], None,
                                                           today_model_file)
        data['msg'] = 'File successfully uploaded'
        resp = jsonify(data)
        return resp
    else:
        data['msg'] = 'Allowed file types are txt, pdf, png, jpg, jpeg, gif'
        resp = jsonify(data)
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)





