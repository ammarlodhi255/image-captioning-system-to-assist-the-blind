from flask import Flask, render_template, request
import os
from predict import *

# Creating flask app
app = Flask(__name__, template_folder='./public')


# Home endpoint, render html file
@app.route('/')
def render():
    return render_template('index.html')

# Run app


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run('0.0.0.0', port=port)
