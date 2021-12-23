import pickle

import joblib
from flask import Flask,render_template,url_for,request



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	feature = open('vec.pkl', 'rb')
	model = open('clf.pkl','rb')

	vec = joblib.load(feature)
	clf = joblib.load(model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = vec.transform(data)
		my_prediction = clf.predict(vect)[0]
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)