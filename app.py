from flask import Flask
from metrics_backend import metrics_bp

app = Flask(__name__, template_folder="templates", static_folder="static")
app.register_blueprint(metrics_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
