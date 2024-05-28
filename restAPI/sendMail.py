import os
from email.message import EmailMessage
import ssl
import smtplib

email_password = 'order2024boost'
email_sender = 'orderboost2024@gmail.com'
email_reciever = 'ninelbenush@gmail.com'

subject = 'Welcome to our website'
body = 'This is the beginning of our work'

server = smtplib.SMTP('smtp.gmail.com',587)
server.starttls()

server.login(email_sender, email_password)

server.sendmail(email_sender,email_reciever, subject)
