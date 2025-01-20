from flask import Flask, jsonify
from noFaceRecogDrowsy import main
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/run-driver-behaviour', methods=['GET'])
def run_driver_behaviour():
    try:
        result = main()  # This will now return a dictionary with warning and physical_state
        warning = result.get("warning", "No warning key found")
        physical_state = result.get("physical_state", "No physical_state key found")

        print(warning)
        print(physical_state)

        # You can return both values in the response as neededs
        return jsonify({
            "message": warning,  # Warning is displayed as message
            "physical_state": physical_state,  # Physical state can be used for bar display
        })
    except Exception as e:
        print(f"Error: {e}")  # Detailed logging
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

#noproblem app.py