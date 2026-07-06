from flask import Flask
from flask_cors import CORS

from routes.pages import pages_bp
from routes.dicom_routes import dicom_bp
from routes.matrix_routes import matrix_bp
from routes.module_classifier_routes import module_classifier_bp


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(pages_bp)
    app.register_blueprint(dicom_bp)
    app.register_blueprint(matrix_bp)

    app.register_blueprint(module_classifier_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

