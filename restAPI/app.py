from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
from file_processor import work_on_file
from flask_mail import Mail, Message
import smtplib

app = Flask(__name__)
CORS(app)

app.config['MAIL_SERVER']='live.smtp.mailtrap.io'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'api'
app.config['MAIL_PASSWORD'] = '06a770732fb9caaaa48f3c7ac08c3031'
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

# Use the correct absolute path to the new database file
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/new_users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Suppress a warning
db = SQLAlchemy(app)
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

def send_mail_for_r(email):  #register
    me = "orderboost2024@gmail.com"
    dest = email

    msg = MIMEMultipart()
    msg['From'] = me
    msg['To'] = dest
    msg['Subject'] = "Thank you for signing up to our service!"

    body = ""
    msg.attach(MIMEText(body, 'plain'))

    img_path = "registerImg.png"
    with open(img_path, "rb") as image_file:
        image = MIMEImage(image_file.read(), name=os.path.basename(img_path))
        msg.attach(image)
    
    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.starttls()
        s.login(me,"nizs kjcb niwc debn")
        s.sendmail(me, dest, msg.as_string())
        s.quit()

def send_predictions(file_path): 
    me = "orderboost2024@gmail.com"
    dest = "ninel.benush@gmail.com"

    msg = MIMEMultipart()
    msg['From'] = me
    msg['To'] = dest
    msg['Subject'] = "Here are your predictions!!"

    body = "Below, you can find the inventory predictions for the upcoming months."
    msg.attach(MIMEText(body, 'plain'))

    filename = os.path.basename(file_path)
    with open(file_path, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={filename}")
        msg.attach(part)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(me, "nizs kjcb niwc debn") 
            s.sendmail(me, dest, msg.as_string())
            s.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")


#@app.route('/', methods=['GET'])


    #s = smtplib.SMTP('smtp.gmail.com', 587)
    #s.starttls()
    #s.login("orderboost2024@gmail.com", "nizs kjcb niwc debn")
    #msg = 'Welcome'
    #s.sendmail("orderboost2024@gmail.com","ninel.benush@gmail.com", msg)
    #s.quit()
    #return jsonify({"message": "Email sent successfully"})

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



#@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "ok"})

@app.route('/',methods=['GET'])
def download_file():
    try:
        file_path = os.path.join(app.root_path, 'files', 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/DataForPrediction/data2.csv')
        if not os.path.isfile(file_path):
            app.logger.error(f"File not found: {file_path}")
            return f"File not found: {file_path}", 404
        send_predictions(file_path)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error sending file: {e}")
        return str(e), 500
    


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

            send_mail_for_r(email)

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
