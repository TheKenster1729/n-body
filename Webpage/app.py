from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def show_page():
    return render_template("./page.html")

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json() # retrieve the data sent from JavaScript
    # process the data using Python code
    result = data['value'] * 2
    return jsonify(result=result) # return the result to JavaScript

if __name__ == "__main__":
    app.run(port = 5000, debug = True)