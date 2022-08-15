import os
import uuid
import ocr
from werkzeug.utils import secure_filename
from flask import Flask, request, abort, jsonify, redirect
from werkzeug.datastructures import ImmutableMultiDict

app = Flask(__name__)

@app.route('/')
def info():
    data = {
        "Name": "Ocr-svc",
        "version": "v1",
        "Author": "Yesvanthraja",
        "Description": "Ocr Utility Service"
    }
    return jsonify(data)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/classify', methods=['POST'])
def upload_file():
    data = {}
    upload_dir = 'uploads/'
    if not os.path.exists(upload_dir):
        print("Uploads directory created")
        os.makedirs(upload_dir)
    print(request.files)
    if 'file' not in request.files:
        file = request.files['file']
        data['msg'] = 'No file part in the request'
        resp = jsonify(data)
        resp.status_code = 400
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
        data['ocr'] = []
        data['ocr'].append({"thresh": ocr.ocr(data['file_name'], data['file_path'], 'thresh'),
                            "blur": ocr.ocr(data['file_name'], data['file_path'], 'blur')})
        data['msg'] = 'File successfully uploaded'
        resp = jsonify(data)
        return resp
    else:
        data['msg'] = 'Allowed file types are  png, jpg, jpeg'
        resp = jsonify(data)
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8082)





