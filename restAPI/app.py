from flask import Flask, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
#CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def root():
    message = "Hello, this is the root route!"
    return jsonify({"message": message})

@app.route("/login", methods=["GET","POST"])
def login():
    return 


if __name__ == '__main__':
    app.run(debug=True)
