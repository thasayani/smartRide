from flask import Flask, jsonify, request
from newDB import main, save_driver_name
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/run-driver-behaviour', methods=['GET'])
def run_driver_behaviour():
    try:
        result = main()  # This will now return a dictionary with warning and physical_state
        warning = result.get("warning", "No warning key found")
        physical_state = result.get("physical_state", "No physical_state key found")
        facename = result.get("facename")

        print(facename)

        print(warning)
        print(physical_state)

        # You can return both values in the response as neededs
        return jsonify({
            "message": warning,  # Warning is displayed as message
            "physical_state": physical_state,  # Physical state can be used for bar display
            "facename": facename
        })
    except Exception as e:
        print(f"Error: {e}")  # Detailed logging
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500


@app.route('/set-user-name', methods=['POST'])
def set_user_name():
    
    try:
        # Get the data from the frontend (user input for name)
        data = request.get_json()

        # Get the name from the frontend input
        user_name = data.get('name')

        if not user_name:
            return jsonify({"error": "Name is required"}), 400
        
        # Save the name to the global variable
        save_driver_name(user_name)

        # Return a success message and updated name to the frontend
        return jsonify({"message": "Name updated successfully", "name": user_name}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
