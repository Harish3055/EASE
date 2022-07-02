from flask import Flask, request, render_template,jsonify
from Barcode import give_me_solution
import cv2
import numpy as np
app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html",text='None')
@app.route('/image', methods=['GET','POST'])
def image():
    i = request.files['image']
    npimg = np.fromfile(i, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    ans = give_me_solution(img)                                
    Process  = [ans.Normal,ans.OTSU_THRESH,ans.CANNY,ans.k_mean_clustered]
    text = 'None'
    for do in Process:
        frame,flag,text = do()
        if flag == 1:
            return jsonify(text)
    return jsonify("Error")
if __name__ == '__main__':
    app.run(port = 5000,  host='0.0.0.0')