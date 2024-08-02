from flask import Flask, request, jsonify
from flask_cors import CORS
from tempfile import NamedTemporaryFile
from deep_translator import GoogleTranslator
import whisper
import os

model = whisper.load_model("base")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def hello():
    return "Welcome to the Whisper API!"

@app.route('/whisper', methods=['POST', 'GET'])
def handler():
    if 'file' not in request.files:
        app.logger.error("File not found in request")
        return jsonify({'error': 'File not found'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    temp = NamedTemporaryFile(delete=False)
    file.save(temp.name)

    try:
        source_language = request.form.get('language', 'english')
        result = model.transcribe(temp.name, language=source_language)
        
        translated = GoogleTranslator(source=source_language, target='english').translate(result['text'])
        response = {'transcript': translated}
       
    except Exception as e:
        app.logger.error(f"Transcription error: {e}")
        response = {'error': 'Failed to transcribe audio'}
        os.close(file)
        os.remove(file)
        return jsonify(response), 500
    finally:
        try:
            if os.path.exists(temp.name):
                os.unlink(temp.name)
                os.close(file)
                os.remove(file)# Clean up the temporary file
        except Exception as e:
            app.logger.error(f"Error deleting temporary file: {e}")

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
