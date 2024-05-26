from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
from file_processor import work_on_file
from flask_mail import Mail, Message

app = Flask(__name__)

#app.config['MAIL_SERVER']= 'live.smtp.mailtrap.io'
#app.config['MAIL_PORT'] = 587
#app.config['MAIL_USERNAME'] = 'api'
#app.config['MAIL_PASSWORD'] = '06a770732fb9caaaa48f3c7ac08c3031'
#app.config['MAIL_USE_TLS'] = True
#app.config['MAIL_USE_SSL'] = False

#mail = Mail(app)

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
    email = db.Column(db.String(200),nullable=False)

    def __init__(self, username, password, email):
        self.username = username
        self.password = password
        self.email = email

@app.before_first_request
def create_tables():
    db.create_all()

#@app.route("/")
#def index():
 #   message = Message(
  #      subject='Thank you for allowing notifications',
   #     recipients=['ninelbenush@gmail.com'],
    #    sender='hi@demomailtrap.com'
    #)
    #message.body = "Hey, welcome to the email notifications, we will send you here all the details"
    #mail.send(message)
    #return "Message sent!"

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "test"})


def create_user(username, password, email):
    new_user = User(username,password,email)
    db.session.add(new_user)
    db.session.commit()


@app.route("/", methods=['POST'])
def handle_post():
    if 'file' in request.files:
        # Handle file upload
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        work_on_file(file_path)

        return f'File {filename} uploaded successfully'
    elif request.is_json:
        # Handle registration or login
        data = request.get_json()
        if 'username' in data and 'password' in data and 'email' in data:
            # Register new user
            username = data['username'].strip()
            password = data['password'].strip()
            email = data['email'].strip()
            user = User.query.filter_by(email=email).first()

            if user:
                app.logger.info(f"Account already exists with this email: {user.email}")
                return jsonify({"message": "Account already exists with this email"}), 409
            
            create_user(username, password, email)
            return jsonify({"message": "User registered successfully"}), 201
        elif 'username' in data and 'password' in data:
            # Login
            username = data['username'].strip()
            password = data['password'].strip()
            user = User.query.filter_by(username=username).first()

            if user and user.password == password:
                message = "Login successful"
                return jsonify({"message": message})
            else:
                return jsonify({"message": "Invalid username or password"}), 401
    return 'Bad Request', 400


if __name__ == '__main__':
    app.run(debug=True)
