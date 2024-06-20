from flask import Flask, render_template, request
import pickle

with open('cancer_death_decisiontree.pkl', 'rb') as f:
    model2 = pickle.load(f)

# Load the second model from the pickle file
with open('cancer_incd_decisiontree.pkl', 'rb') as f:
    model1 = pickle.load(f)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('welcome.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/inc')
def inc():
    return render_template('incd.html', predicted_value='')
@app.route('/dea')
def dea():
    return render_template('death.html', predicted_value='')
@app.route('/death', methods=['POST'])
def death():
    # Retrieve the form data
   
    fips = request.form['fips']
    abc = request.form['abc']
    lower = request.form['lower']
    upper = request.form['upper']
    avg = request.form['avg']
    rate = request.form['rate']
    lowconf = request.form['lowconf']
    upconf = request.form['upconf']
    metobj = request.form['metobj']
    
        

    # Perform your prediction or desired processing
    a=model2.predict([[fips,abc,lower,upper,avg,rate,lowconf,upconf,metobj,0.0]])


    # Render the template with the predicted value
    return render_template('deathres.html', predicted_value=a)

@app.route('/incd', methods=['POST'])
def incd():
    # Retrieve the form data
    
    fips = request.form['fips']
    abc = request.form['abc']
    lower = request.form['lower']
    upper = request.form['upper']
    avg = request.form['avg']
    rate = request.form['rate']
    lowconf = request.form['lowconf']
    upconf = request.form['upconf']

    # Process the form data or perform your desired actions here
    a=model1.predict([[fips,abc,lower,upper,avg,rate,lowconf,upconf]])
    # Render a response or redirect to another page
    return render_template('incdres.html', predicted_value=a)

if __name__ == '__main__':
    app.run()