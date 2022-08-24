from flask import *  
from model import model
import sys
sys.path.append("..")

app = Flask(__name__)
depPredict = model.DepPredict()
# print(depPredict.prediction("1","2","3"))  

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

        nii.save("./model/" + nii.filename)  
        bval.save("./model/" + bval.filename)
        bvec.save("./model/" + bvec.filename)
        output = depPredict.prediction(nii.filename, bval.filename, bvec.filename)
        print(output)
        status = "NO"
        if output[0] == 1:
            status = "YES"    

        return jsonify({
            "output":status,
        }) 




if __name__ == '__main__':  
    app.run(host= '0.0.0.0', debug = True)  
