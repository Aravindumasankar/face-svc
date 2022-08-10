from crypt import methods
import json
from flask import Flask, request, abort, jsonify
from flask_restplus import Api, Resource, fields
from functools import wraps


flask_app = Flask(__name__)
restful_app = Api(app = flask_app,
          version = "1.0", 
		  title = "Face Utility", 
		  description = "Microservice for Face Utilities.")
name_space = restful_app.namespace('face', description='Face API\'s')


# defining the APIs

# def get(self):
#     		return {
# 			"status": "Got new data"
#             }

@name_space.route("/", methods = ['GET'])
class MainClass(Resource):
    @restful_app.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error' }, 
			 params={ 'key': 'Specify the key associated with the face image' })
    def get(self):
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
        return {
            "status" : "Data recieved"
        }

# model = restful_app.model('Face Model', 
# 		            {'image': fields.String(required = True, 
# 					 description="Face of the person")})

# @name_space.route("/classify", methods = ['POST'])
# class MainClass(Resource):
# 	@restful_app.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error' }, 
# 			 params={ 'key': 'Specify the key associated with the face image' })
# 	@restful_app.expect(model)		
# 	def post(self, key):
# 		try:
# 			face_list[key] = request.json['image']
# 			return {
# 				"status": "New person image added",
# 				"face_key": face_list[key]
# 			}
# 		except KeyError as e:
# 			name_space.abort(500, e.__doc__, status = "Could not save information", statusCode = "500")
# 		except Exception as e:
# 			name_space.abort(400, e.__doc__, status = "Could not save information", statusCode = "400")

	# def post(self):
	# 	return {
	# 		"status": "Posted new data"
	# 	}




# @flask_app.route('/classify', methods=['POST'])
# @require_appkey
# def classify():
#     data = {}
#     data['msg'] = 'Unsupported Format. Required .png, .jpg, .jpeg'
#     resp = jsonify(data)
#     resp.status_code = 401
#     return resp


if __name__ == "__main__":
    flask_app.run(host='0.0.0.0', port=8080)
