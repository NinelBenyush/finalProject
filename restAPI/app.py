from flask import Flask, request, jsonify
#from flask_cors import CORS


app = Flask(__name__)
#CORS(app)
#CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def root():
    message = "test"
    return jsonify({"message": message})

@app.route("/login", methods=["GET","POST"])
def login():
    username = request.json['username']
    password = request.json['password']
    name = request.json['name']
    print(username + " " + password + " " + name)
    message = "test"
    return jsonify({"message": message})


if __name__ == '__main__':
    app.run(debug=True)
