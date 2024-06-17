from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pyexpat.errors import messages
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
from sqlalchemy import text
import requests
from file_processor import work_on_file
from flask_mail import Mail, Message
import smtplib
import datetime



app = Flask(__name__)
CORS(app)


mail = Mail(app)

# Use the correct absolute path to the new database file
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/new_users.db'
app.config['SQLALCHEMY_BINDS'] = {
    'files_db': 'sqlite:///C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/uploaded_files.db',
    'results_db': 'sqlite:///C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/results.db',
    'basicInfo_db':'sqlite:///C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/basicInfo.db'
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Suppress a warning
db = SQLAlchemy(app)
UPLOAD_FOLDER = "./UPLOAD_FOLDER"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
FILE_DIRECTORY = 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/DataForPrediction'

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

class UploadFiles(db.Model):
    __bind_key__ = 'files_db' 
    __tablename__ = 'UploadFiles'
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(400), nullable=False)

    def __init__(self, file_name, description):
        self.file_name = file_name
        self.description = description

class Result(db.Model):
    __bind_key__ = 'results_db' 
    __tablename__ = 'results'
    id = db.Column(db.Integer, primary_key=True)
    fileName = db.Column(db.String(100), nullable=False)
    date = db.Column(db.DateTime) 

    def __init__(self, fileName, date):
        self.fileName = fileName
        self.date = date

def save_to_database(filename, upload_time):
    result = Result(fileName=filename, date=upload_time)
    db.session.add(result)
    db.session.commit()


class BasicInfo(db.Model):
    __bind_key__ = 'basicInfo_db' 
    __tablename__ = 'basicInfo'
    id = db.Column(db.Integer, primary_key=True)
    firstName = db.Column(db.String(100), nullable=False)
    lastName = db.Column(db.String(100), nullable=False)
    companyName =  db.Column(db.String(100), nullable=False)
    phoneNumber = db.Column(db.String(20), nullable=False, unique=True)
    companyDescription = db.Column(db.String(100), nullable=False)
    emailAddress = db.Column(db.String(100))

    def __init__(self, firstName, lastName, companyName, phoneNumber, companyDescription,emailAddress):
        self.firstName = firstName
        self.lastName = lastName
        self.companyName = companyName
        self.phoneNumber = phoneNumber
        self.companyDescription = companyDescription
        self.emailAddress = emailAddress
        

@app.before_first_request
def create_tables():
    db.create_all()
    with app.app_context():
        bind_key_files = 'files_db'
        query_files = text("CREATE TABLE IF NOT EXISTS file_details (id INTEGER NOT NULL, filename VARCHAR(200) NOT NULL, description VARCHAR(500) NOT NULL, PRIMARY KEY (id));")
        db.session.execute(query_files)
        
        bind_key_results = 'results_db'
        query_results = text("CREATE TABLE IF NOT EXISTS results (id INTEGER NOT NULL, fileName VARCHAR(100) NOT NULL,date DATETIME, PRIMARY KEY (id));")
        db.session.execute(query_results)

        bind_key_info = 'basicInfo_db'
        query_info = text("""
                    CREATE TABLE IF NOT EXISTS basicInfo (
                    id INTEGER NOT NULL PRIMARY KEY,
                    firstName VARCHAR(100) NOT NULL,
                    lastName VARCHAR(100) NOT NULL,
                    companyName VARCHAR(100) NOT NULL,
                    phoneNumber VARCHAR(20) NOT NULL UNIQUE,
                    companyDescription VARCHAR(100) NOT NULL,
                    emailAddress VARCHAR(100)
                    );
                """)
        db.session.execute(query_info)

        db.session.commit()
        


def send_mail_for_reminder():
    me = "orderboost2024@gmail.com"
    dest = "ninel.benush@gmail.com"

    msg = MIMEMultipart()
    msg['From'] = me
    msg['To'] = dest
    msg['Subject'] = "Reminder!"

    date = datetime.datetime.now()
    new_date = date + datetime.timedelta(days=6)
    day = new_date.day
    month = new_date.month
    year = new_date.year

    body = f"Don't forget to make the payment by {day}/{month}/{year}"
    msg.attach(MIMEText(body, 'plain'))

    img_path = "Reminder.png"
    with open(img_path, "rb") as image_file:
        image = MIMEImage(image_file.read(), name=os.path.basename(img_path))
        msg.attach(image)
    
    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.starttls()
        s.login(me,"nizs kjcb niwc debn")
        s.sendmail(me, dest, msg.as_string())
        s.quit()

    reminder_message = {
        "content": body,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d")
    }
    try:
        response = requests.post("http://localhost:5000/profile/updates", json=reminder_message)
        response.raise_for_status()  # Raise an error for bad status codes
        print("Reminder message sent to server successfully")
    except requests.exceptions.RequestException as e:
        print(f"Error sending reminder message to server: {e}")
    return reminder_message



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


def send_email_for_upload(filename):
    me = "orderboost2024@gmail.com"
    email = "forpracticepython2023@gmail.com"
    dest = email

    msg = MIMEMultipart()
    msg['from'] = me
    msg['To'] = dest
    msg['Subject'] = f'You upload the file {filename} succesfully'

    body = "We have received your file, and you will soon receive the results you are waiting for."
    msg.attach(MIMEText(body, 'plain'))

    img_path = "upload.png"
    with open(img_path, "rb") as image_file:
        image = MIMEImage(image_file.read(), name=os.path.basename(img_path))
        msg.attach(image)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(me, "nizs kjcb niwc debn")
            s.sendmail(me, dest, msg.as_string())
            s.quit()
            print("email send succesfully")
    except Exception as e:
        print(f"error {e}")



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



@app.route('/', methods=['GET'])
def root():
    send_mail_for_reminder()
    return jsonify({"message": "ok"})

@app.route('/profile/updates', methods=['GET'])
def get_updates():
    # Implement logic to retrieve and return messages
    return jsonify([send_mail_for_reminder()]),200


@app.route('/profile/updates', methods=['POST'])
def profile_updates():
    reminder_message = request.json
    print("message:", reminder_message)
    # Add the reminder_message to your database or process it as needed
    return jsonify({"message": "Reminder message received"}), 201


def create_user(username, password, email):
    new_user = User(username,password,email)
    db.session.add(new_user)
    db.session.commit()

def create_new_basic_info(fName, lName, cName, phoneNumber, cDescription,emailAddress):
    new_info = BasicInfo(fName, lName, cName, phoneNumber, cDescription,emailAddress)
    db.session.add(new_info)
    db.session.commit()

uploaded_files = []
latest_res = {}
@app.route("/upload-file", methods=['POST'])
def handle_post():
    global latest_res 
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400

        filename = file.filename
        send_email_for_upload(filename=filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        upload_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #print(upload_time)

        res_file_path = work_on_file(file_path)
        res_filename = os.path.basename(res_file_path)
        res_time = datetime.datetime.now()

        
        save_to_database(res_filename, res_time)
        uploaded_files.append({'fileName': filename, 'upload_time': upload_time})

        response = {
            'message ':"File uploaded  successfully",
            'another_m' :f'File {filename} uploaded successfully',
            'file_name' : filename,
            'upload_time': upload_time,
            "final_m" : "You got the results, check in the results section",
        }
        latest_res = response

        return jsonify(response)
    return 'Bad Request', 400

@app.route("/get-res", methods=['GET'])
def get_res():
    global latest_res
    res_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    response = {
            
                "message": latest_res.get("final_m", ""),
                "name": latest_res.get("file_name", ""),
                "res_time": res_time
            
    }
    return jsonify(response)

@app.route("/uploaded-files", methods=['GET'])
def get_latest_upload():
    response = {
        'status': 'success',
        'results': uploaded_files
    }
    return jsonify(response), 200


@app.route("/register", methods=['POST'])
def handle_register():
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

        #send_mail_for_r(email)

        return jsonify({"message": "User registered successfully"}), 201

login_info = []
@app.route("/login", methods=["POST"])
def handle_login():
    data = request.get_json()
    if 'username' in data and 'password' in data:
        username = data['username'].strip()
        password = data['password'].strip()
        user = User.query.filter_by(username=username).first()
        
        if user and user.password == password:
            response = {
               " message ":"Login successful",
                 "notification":"Welcome back"
            }
            login_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            login_info.append({'username': username, 'login_time': login_time})
            
            return jsonify(response)
        else:
            return jsonify ({"message": "Invalid username or password"}), 401
        
@app.route("/get-login", methods=['GET'])
def get_welcome():
    response = {
        'status': 'success',
        'results': login_info
    }
    return jsonify(response), 200


@app.route("/download-file", methods=['GET'])
def download_file():
    try:
        file_path = os.path.join(app.root_path, 'files', 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/DataForPrediction/data.csv')
        if not os.path.isfile(file_path):
            app.logger.error(f"File not found: {file_path}")
            return f"File not found: {file_path}", 404
        download_time = datetime.datetime.now().strftime('%Y-%m-%d')
        send_predictions(file_path)
        return send_file(file_path, as_attachment=True, mimetype='text/csv')
    except Exception as e:
        app.logger.error(f"Error sending file: {e}")
        return str(e), 500
    
@app.route("/profile", methods=['POST'])
def confirm_new_file():
    data = request.json
    if 'filename' in data and 'description' in data:
        filename = data['filename']
        description = data['description']
        new_file = UploadFiles(filename, description)
        db.session.add(new_file)
        db.session.commit()
        response = {
            'status': 'success',
            'message': 'File details received',
            'filename': filename,
            'description': description
        }
        return jsonify(response), 201
    else:
        response = {
            'status': 'error',
            'message': 'Missing filename or description'
        }
        return jsonify(response), 400
    
@app.route("/profile/files", methods=['GET'])
def get_files():
    files = UploadFiles.query.all()
    file_list = [{"filename": file.file_name, "description": file.description} for file in files]
    response = {
        'status': 'success',
        'files': file_list
    }

    return jsonify(response), 200

@app.route("/profile", methods=['GET'])
def showPersonalInfo():
    infos = BasicInfo.query.all()
    info_list = [
        {
            'firstName': info.firstName,
            'lastName': info.lastName,
            'companyName': info.companyName,
            'phoneNumber': info.phoneNumber,
            'companyDescription': info.companyDescription,
            'emailAddress': info.emailAddress
        }
        for info in infos
    ]
    
    response = {
        'status': 'success',
        'info': info_list
    }
    
    return jsonify(response), 200



downloaded_files = []
@app.route("/profile/results", methods=['GET'])
def get_result():
    results = Result.query.all()
    result_list =  [{"filename": file.fileName, 'date':file.date} for file in results]
    response = {
        'status':'success',
        'results':result_list
    }
    
    return jsonify(response), 200

RESULTS_FOLDER = "C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/results"
@app.route("/profile/results/<filename>.csv", methods=['GET'])
def download_right_file(filename):
    print(filename)
    file_path = os.path.join(RESULTS_FOLDER, filename) 
    file_path = os.path.normpath(file_path)  
    file_path = file_path.replace('\\', '/') 
    print(file_path)
    if os.path.exists(file_path):
        download_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        downloaded_files.append({'filename': filename, 'download_time': download_time})
        return send_from_directory(directory=FILE_DIRECTORY, path=filename, as_attachment=True), 201
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404

@app.route("/getDownload", methods=['GET'])
def getDInfo():
    response = {
        'status':'success',
        'results':downloaded_files,
    }
    return jsonify(response),200

@app.route("/basic-info", methods=['POST'])
def handle_basic_info():
    data = request.get_json()
    fName = data['fName'].strip()
    lName = data['lName'].strip()
    cName = data['cName'].strip()
    phoneNumber = data['phoneNumber'].strip()
    cDescription = data['cDescription'].strip()
    email = data['email'].strip()
    create_new_basic_info(fName, lName, cName, phoneNumber,cDescription,email)

    return jsonify({"message": "basic info updated successfully"}), 201


if __name__ == '__main__':
    app.run(debug=True)
