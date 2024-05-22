from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os

app = Flask(__name__)

# Use the correct absolute path to the new database file
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/new_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Suppress a warning
db = SQLAlchemy(app)
CORS(app)
UPLOAD_FOLDER = "./UPLOAD_FOLDER"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class User(db.Model):
    __tablename__ = 'users'  # Ensure the table name matches
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@app.before_first_request
def create_tables():
    db.create_all()

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "test"})

@app.route("/", methods=['POST'])
def handle_post():
    if 'file' in request.files:
        file = request.files['file']

        if file.filename == '':
            return 'No file selected', 400


        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        return f'File {filename} uploaded successfully'

    elif request.is_json:
        # Handle login
        try:
            data = request.get_json()
            username = data['username'].strip()
            password = data['password'].strip()

            app.logger.info(f"Received username: '{username}', password: '{password}'")

            user = User.query.filter_by(username=username).first()

            if user:
                app.logger.info(f"Retrieved user: id={user.id}, username={user.username}, password={user.password}")
            else:
                app.logger.info(f"No user found with the provided username '{username}'")

            if user and user.password == password:
                message = "Login successful"
                return jsonify({"message": message})
            else:
                return jsonify({"message": "Invalid username or password"}), 401
        except Exception as e:
            app.logger.error(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), 400

    return 'Bad Request', 400
if __name__ == '__main__':
    app.run(debug=True)
