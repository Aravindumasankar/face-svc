from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    data = {
        'name': 'face-svc',
        'version': 'v1',
        'purpose': ['Face Detection', 'Face Recognition', 'Emotion Recognition'],
        'author': 'aravind-umasankar'
            }
    resp = jsonify(data)
    resp.status_code = 200
    return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
