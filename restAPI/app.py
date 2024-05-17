from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
CORS(app)
#CORS(app, resources={r"/*": {"origins": "*"}})

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)  


@app.route('/', methods=['GET'])
def root():
    message = "test"
    return jsonify({"message": message})

@app.route("/", methods=["POST"])
def login():
    try:
        data = request.get_json()  # Automatically handles the Content-Type check
        username = data['username']
        password = data['password']
        print(f"{username} {password}")
        message = username
        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)
