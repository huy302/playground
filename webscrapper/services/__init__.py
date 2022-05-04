from flask import Flask, jsonify, request
from flask_restx import Api
import sys

def create_app(env=None):
    from services.config import config_by_name
    from services.routes import register_routes

    app = Flask(__name__)
    app.config.from_object(config_by_name[env or "test"])
    
    api = Api(app, title="Powertrain Automated Services", version="0.0.1", description=(
        "Automated service interacting with Powertrain Web UI"
    ))

    register_routes(api, app)

    @app.route("/health")
    def health():
        return jsonify("Server healthy")
    
    @app.route("/mei")
    def mei():
        if hasattr(sys, '_MEIPASS'):
            return jsonify(sys._MEIPASS)
        return jsonify("")

    @app.route('/shutdown')
    def shut_down():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()
        return jsonify("Server terminated")

    return app