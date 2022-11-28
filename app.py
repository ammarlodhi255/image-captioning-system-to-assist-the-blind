from flask import Flask, render_template, request

# Creating flask app
app = Flask(__name__, template_folder='./public')


@app.route('/')
def render():
    return render_template('index.html')
