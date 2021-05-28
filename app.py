import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = [ "texture_mean", "perimeter_mean","area_mean","smoothness_mean", "fractal_dimension_mean", "radius_se",
                       "texture_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se","symmetry_se",
                        "fractal_dimension_se","texture_worst","perimeter_worst	","area_worst", ]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 1:
        res_val = ".....  breast cancer    ..... "
    else:
        res_val = ".....  no Breast cancer    .... "


    return render_template('index.html', prediction_text=' Patient have {}'.format(res_val))

if __name__ == "__main__":
    app.run()
