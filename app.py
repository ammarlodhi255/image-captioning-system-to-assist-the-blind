from flask import Flask, render_template, request
from predict import *


app = Flask(__name__, template_folder='./public')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


# Initialize the model
model = init_model()


# Home endpoint, render html file
@app.route('/')
def render():
    return render_template('index.html')


def speak_caption(caption):
    text_to_speech = gTTS(caption)
    text_to_speech.save('./static/speech.wav')
    file = r'D:\University Files\Assignments\7th Semester\Machine Learning\Project\source-code\static\speech.wav'
    Audio(file, autoplay=True)


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    global model
    file = request.files['file1']
    file.save('./static/file.jpg')

    predicted_caption = predict_caption(model, './static/file.jpg')
    # speak_caption(predicted_caption)
    return render_template('index.html', predicted_caption=predicted_caption)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run('0.0.0.0', port=port)
