from flask import Flask, render_template, request, jsonify
import model

app = Flask(__name__)

@app.route("/")
def index():
    status = model.check_status()
    return render_template("index.html", status=status)

@app.route("/api/status", methods=["GET"])
def check_status():
    status = model.check_status()
    return jsonify(status)

@app.route("/api/train", methods=["POST"])
def train_model():
    try:
        data_res = model.create_dataset()
        train_res = model.train()
        return jsonify({
            "status": "success",
            "message": "Models trained successfully!",
            "details": train_res
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"status": "error", "message": "No text provided"}), 400
        
    result = model.predict_news(text)
    if result["status"] == "error":
        return jsonify(result), 400
        
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
