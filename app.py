#Creating a flask web application

import pickle #it is needed to load the regression model pickle file
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__) # __name__ is starting point of my application from where it will run
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
# this localhost the url and slash if i basically say i should go to the home page
@app.route('/') 
# function to redirect to the home.html file
def home(): 
# i.e. basically when we hit the flask app it goes to redirect the home page    
    return render_template('home.html') #render_template redirect to the template folder

# now we are going to make sure that we create a predict api so for creating predict api i am just create an api and using postman we can send the request to our app and get the output
@app.route('/predict_api',methods=['POST']) 
# from here as soon as I hit the api as a post request with 'data' info the information is captured by using request.json and then this will be stored in data variable 
def predict_api():
    # the input i going to give which i make sure that i give in json format which is captured inside data key
    data=request.json['data']
    print(data)
#when we get the data from json we get in key value pairs
 #we convert it into list and after getting the pickle file we have to reshape it#
    print(np.array(list(data.values())).reshape(1,-1))
    #(1,-1) because it is a single data point record that we get and take it to do the transformation and it want single record record with so many number of features based on our data set#
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0]) #since it is a 2D array so i am taking the first value 
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()] #imported from flask(⭐whatever values we fill in this form it automatically capture it fro the request object )
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    #this below renders in html and in this html file there may be some placeholder so it replace it with this placeholder
    return render_template("home.html",prediction_text="The predicted House price is {}".format(output)) # the position where prediction_text is written in html file it prints the output there
#for running it
if __name__=="__main__":
    app.run(debug=True) #it runs in debug mode 
