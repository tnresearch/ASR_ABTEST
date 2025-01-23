from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import json
import requests
from datetime import datetime
from pathlib import Path
import wave
from asr_abtest.benchmark.batch_runner import BatchRunner
from asr_abtest.benchmark.results import BenchmarkResults
import argparse

app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')

# Ensure upload directories exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print(f"Template directory: {os.path.join(os.path.dirname(__file__), 'templates')}")

def count_tokens(text):
    """Simple word-based token counter"""
    return len(text.split())

def get_wav_duration(wav_path):
    """Get duration of wav file in seconds"""
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Warning: Could not read WAV duration: {e}")
        return None  # Return None if we can't read the duration

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'audio' not in request.files or 'transcript1' not in request.files or 'transcript2' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    audio_file = request.files['audio']
    transcript1_file = request.files['transcript1']
    transcript2_file = request.files['transcript2']
    
    if audio_file.filename == '' or transcript1_file.filename == '' or transcript2_file.filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    if not audio_file.filename.endswith('.wav'):
        return jsonify({'error': 'Audio must be WAV format'}), 400
    
    if not transcript1_file.filename.endswith('.json') or not transcript2_file.filename.endswith('.json'):
        return jsonify({'error': 'Transcripts must be JSON format'}), 400
    
    # Save files
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    transcript1_path = os.path.join(UPLOAD_FOLDER, transcript1_file.filename)
    transcript2_path = os.path.join(UPLOAD_FOLDER, transcript2_file.filename)
    
    audio_file.save(audio_path)
    transcript1_file.save(transcript1_path)
    transcript2_file.save(transcript2_path)
    
    print(f"Saved audio file to: {audio_path}")
    print(f"File exists: {os.path.exists(audio_path)}")
    print(f"File size: {os.path.getsize(audio_path)} bytes")
    
    return jsonify({
        'success': True,
        'audio': audio_file.filename,
        'transcript1': transcript1_file.filename,
        'transcript2': transcript2_file.filename
    })

@app.route('/assets/<path:filename>')
def serve_file(filename):
    response = send_from_directory(UPLOAD_FOLDER, filename)
    
    # Set correct MIME type for WAV files
    if filename.endswith('.wav'):
        response.headers['Content-Type'] = 'audio/wav'
    
    # Add debug headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    print(f"Serving file: {filename} with type: {response.headers['Content-Type']}")
    return response

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not audio_file.filename.endswith('.wav'):
        return jsonify({'error': 'Audio must be WAV format'}), 400
    
    try:
        # Save temporary file
        audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(audio_path)
        
        # Call the Whisper API
        with open(audio_path, 'rb') as f:
            response = requests.post('http://localhost:8000/transcribe', files={'file': f})
        
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                'success': True,
                'words': data['words']
            })
        else:
            return jsonify({'error': 'Transcription failed'}), 500
            
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ui/config.json')
def serve_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'comparison_config.json')
    try:
        with open(config_path, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        print(f"Error serving config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_metadata', methods=['POST'])
def save_metadata():
    try:
        data = request.json
        audio_file = data['audio']
        transcript1_file = data['transcript1']
        transcript2_file = data['transcript2']
        wer = data.get('wer', {})
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "comparison_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(results_dir, f"response_{timestamp}.json")
        
        # Get audio file metadata
        audio_path = os.path.join(UPLOAD_FOLDER, audio_file)
        audio_size = os.path.getsize(audio_path)
        audio_duration = get_wav_duration(audio_path)  # This might return None now
        
        # Create metadata object
        metadata = {
            "timestamp": timestamp,
            "audio": {
                "filename": audio_file,
                "duration_seconds": audio_duration,  # This might be None
                "filesize_bytes": audio_size
            },
            "transcripts": {
                "transcript1": {
                    "filename": transcript1_file,
                    "num_tokens": count_tokens(' '.join([w['text'] for w in json.load(open(os.path.join(UPLOAD_FOLDER, transcript1_file), 'r'))['words']]))
                },
                "transcript2": {
                    "filename": transcript2_file,
                    "num_tokens": count_tokens(' '.join([w['text'] for w in json.load(open(os.path.join(UPLOAD_FOLDER, transcript2_file), 'r'))['words']]))
                }
            },
            "wer_scores": wer,
            "ratings": None
        }
        
        # Save to file
        with open(result_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return jsonify({
            'success': True,
            'result_file': result_file
        })
        
    except Exception as e:
        print(f"Error saving metadata: {e}")
        import traceback
        traceback.print_exc()  # This will print the full error trace
        return jsonify({'error': str(e)}), 500

@app.route('/save_ratings', methods=['POST'])
def save_ratings():
    try:
        data = request.json
        result_file = data['result_file']
        ratings = data['ratings']
        
        # Read existing metadata
        with open(result_file, 'r') as f:
            metadata = json.load(f)
        
        # Update with ratings
        metadata['ratings'] = ratings
        
        # Save updated metadata
        with open(result_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error saving ratings: {e}")
        return jsonify({'error': str(e)}), 500

# Add a new route to serve response files
@app.route('/results/<path:filename>')
def serve_result(filename):
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    try:
        return send_from_directory(results_dir, filename)
    except Exception as e:
        print(f"Error serving result file: {e}")
        return jsonify({'error': str(e)}), 404

batch_runner = BatchRunner()
benchmark_results = BenchmarkResults()

@app.route('/benchmark/process', methods=['POST'])
async def process_benchmark():
    try:
        files = []
        for audio, reference in zip(
            request.files.getlist('audio_files[]'),
            request.files.getlist('reference_files[]')
        ):
            files.append({
                'audio': {
                    'filename': audio.filename,
                    'content': audio.read()
                },
                'truth': {
                    'filename': reference.filename,
                    'content': reference.read()
                }
            })
        
        model_id = request.form['model_id']
        results = await batch_runner.run_batch(files, model_id)
        
        # Save results
        results_file = benchmark_results.save_results(results)
        
        return jsonify({
            'success': True,
            **results,
            'results_file': results_file
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def main():
    parser = argparse.ArgumentParser(description='Start ASR UI')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port to bind to')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main() 