from flask import Flask, render_template, request
import os

# Creating flask app
app = Flask(__name__, template_folder='./public')


# Home endpoint, render html file
@app.route('/')
def render():
    return render_template('index.html')

# Run app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))