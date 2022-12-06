
from flask import Flask, render_template, request
from predict import *
from tts import *

app = Flask(__name__, template_folder='./public')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


# Initialize the model
model = init_model()
predicted_text = None

# Home endpoint, render html file
@app.route('/')
def render():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    global model
    file = request.files['file1']
    file.save('./static/file.jpg')

    predicted_caption = predict_caption(model, './static/file.jpg')
    predicted_text = predict_caption
    text_to_speech(predicted_caption, "Male")
    predicted_caption = "Generated caption:\n" + predicted_caption
    return render_template('predicted.html', predicted_caption=predicted_caption)

@app.route('/speak')
def render():
    if predicted_text == None:
        text_to_speech('Sorry! Invalid Request', "Male")
        return
    text_to_speech(predicted_text, "Male")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run('0.0.0.0', port=port)
