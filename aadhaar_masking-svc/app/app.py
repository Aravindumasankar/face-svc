import os
import uuid
import aadhar_masking
from werkzeug.utils import secure_filename
from flask import Flask, request, abort, jsonify, redirect, send_file
from werkzeug.datastructures import ImmutableMultiDict

app = Flask(__name__)

@app.route('/')
def info():
    data = {
        "Name": "Aadhar Masking-svc",
        "version": "v1",
        "Author": "Yesvanthraja",
        "Description": "Aadhar Utility Service"
    }
    return jsonify(data)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','pdf'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/<dir_name>/<path:filename>')

def download_file(dir_name, filename):

    file_path =  dir_name +'\\'+ filename

    print(file_path)

    return send_file(file_path)


@app.route('/classify', methods=['POST'])
def upload_file():
    data = {}
    upload_dir = 'uploads/'
    if not os.path.exists(upload_dir):  
        print("Uploads directory created")
        os.makedirs(upload_dir)
    processed_dir = 'processed/'
    if not os.path.exists(processed_dir):  
        print("Processed directory created")
        os.makedirs(processed_dir)
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
        data['uploaded_file_path'] = str(request.url_root+upload_dir + data['file_name'])
        data['processed_filepath'] = str(request.url_root+processed_dir + data['file_name'])
        data['masked'] = aadhar_masking.mask_coordinates(data['file_name'], data['file_path'], 50)
        data['msg'] = 'File successfully uploaded'
        resp = jsonify(data)
        return resp
    else:
        data['msg'] = 'Allowed file types are  png, jpg, jpeg, pdf'
        resp = jsonify(data)
        resp.status_code = 400
        return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3036)