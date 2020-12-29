import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from numpy import inf

app = Flask(__name__,static_url_path = "/tmp", static_folder = "tmp")
rf_model = pickle.load(open('bolt_r_rf.pkl', 'rb'))
ab_model = pickle.load(open('bolt_r_ab.pkl', 'rb'))
lr_model = pickle.load(open('bolt_r_lr.pkl','rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    
    #print(int_features)
    e1_do=int_features[0]/int_features[2]
    e2_do=int_features[1]/int_features[2]
    fu_fy = int_features[3]/int_features[4]
    type_c = int_features[5]
    #print(e1_do,e2_do,fu_fy,type_c)
    #print(int_features[5])
    final_features=[]
    
    #final_features=final_features+[np.log(e2_do)]
    #final_features=final_features+[np.log(fu_fy)]
    
    if e1_do!=0:
        final_features=final_features+[np.log(e1_do)]
    else:
        final_features=final_features+[(e1_do)]
        
    if e2_do!=0:
        final_features=final_features+[np.log(e2_do)]
    else:
        final_features=final_features+[(e2_do)]
        
    if fu_fy!=0:
        final_features=final_features+[np.log(fu_fy)]
    else:
        final_features=final_features+[(fu_fy)]
        
        
    if type_c!=0:
        final_features=final_features+[np.log(type_c)]
    else:
        final_features=final_features+[type_c]
    
    
    
    final_features = [np.array(final_features)]
    print(final_features)
    

    #final_features=np.log(final_features)

    #print(final_features)
    final_features=scaler.transform(final_features)
    
    
    print(final_features)
    
    
    rf_prediction = rf_model.predict(final_features)
    ab_prediction = ab_model.predict(final_features)
    lr_prediction = lr_model.predict(final_features)
    

    
    

    rf_prediction_o = round(rf_prediction[0], 3)
    ab_prediction_o = round(ab_prediction[0], 3)
    lr_prediction_o = round(lr_prediction[0], 3)
    
    
    
    rf_prediction_o=format(rf_prediction_o,'.3f')
    ab_prediction_o=format(ab_prediction_o,'.3f')
    lr_prediction_o=format(lr_prediction_o, '.3f')
    
    
    print(rf_prediction_o,ab_prediction_o,lr_prediction_o)
    

    
    return render_template('index.html', rf='{}'.format(rf_prediction_o), ab='{}'.format(ab_prediction_o), lr='{}'.format(lr_prediction_o)) 


if __name__ == "__main__":
    app.run(debug=True)