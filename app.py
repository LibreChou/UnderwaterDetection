from flask import Flask
app = Flask(__name__)

from detector import Detector

detect= Detector()

@app.route('/detete')
def detect():
    img = None
    boxes =detect(img)

    return boxes

if __name__ == '__main__':
   app.run()