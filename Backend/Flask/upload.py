from flask import *  
from model import model
import sys
import os
from flask_cors import CORS
sys.path.append("..")

app = Flask(__name__)
CORS(app)
depPredict = model.DepPredict()
cnn = model.CNN()
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)  
        return render_template("success.html", name = f.filename)

@app.route('/evaluate', methods = ['POST'])
def evaluate():
    if request.method == 'POST':  
        nii = request.files['nii']  
        bval = request.files['bval']  
        bvec = request.files['bvec'] 
        print(nii,bval,bvec)
        nii.save("./model/" + nii.filename)  
        bval.save("./model/" + bval.filename)
        bvec.save("./model/" + bvec.filename)
        output = depPredict.prediction(nii.filename, bval.filename, bvec.filename)
        print(output)
        if(output[0][0]>output[0][1]):
            score = output[0][0]*100
            status = "NO"
        else:
            score = output[0][1]*100
            status = "YES"

        return jsonify({
            "score":score,
            "stat":status
        }) 

@app.route('/carousel', methods = ['GET'])
def carousel():
        files= [f"{request.form['file']}_{x}_3" for x in [0,2,4,7]]
        for file in files:
            if not os.path.exists('static/images/'+file+".png"):
                return jsonify({"output":"Carousel Images not found"})  
        files = [url_for('static', filename="images/"+file+".png") for file in files]

        quads = cnn.depression_quadrant(request.form['file'])

        return jsonify({"output":files,"quads":quads})   

if __name__ == '__main__':  
    app.run(host= '127.0.0.1', debug = True)  
