from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hola'
    #return render_template('Index.html')

@app.route('/about')
def about():
    return 'Program'

if __name__ == '__main__':
    app.run()