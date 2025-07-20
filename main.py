import subprocess  # Standard library
from flask import Flask, request  # Third-party libraries
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes (helpful for Replit/n8n/frontend triggering)
CORS(app)

@app.route('/')
def index():
    return "Agentic Workflow Trigger is live!"

@app.route('/run_blog', methods=['POST'])
def run_blog():
    """Trigger blog content workflow"""
    subprocess.Popen(["python", "combined_blog_workflow.py"])
    return "Blog workflow started", 200

@app.route('/run_reviews', methods=['POST'])
def run_reviews():
    """Trigger product reviews workflow"""
    subprocess.Popen(["python", "combined_reviews_workflow.py"])
    return "Reviews workflow started", 200

@app.route('/run_press_release', methods=['POST'])
def run_press_release():
    """Trigger press release workflow"""
    subprocess.Popen(["python", "combined_press_release_workflow.py"])
    return "Press release workflow started", 200

@app.route('/run_case_study', methods=['POST'])
def run_case_study():
    """Trigger case study workflow"""
    subprocess.Popen(["python", "combined_case_study_workflow.py"])
    return "Case study workflow started", 200

@app.route('/run_video_transcript', methods=['POST'])
def run_video_transcript():
    """Trigger video transcript workflow"""
    subprocess.Popen(["python", "combined_video_transcript_workflow.py"])
    return "Video transcript workflow started", 200

@app.route('/run_standard_analysis', methods=['POST'])
def run_standard_analysis():
    """Trigger standard question analysis to cards workflow"""
    subprocess.Popen(['python', 'standard_question_analyzer_to_cards02.py'])
    return "Standard analysis workflow started", 200

@app.route('/run_video_script', methods=['POST'])
def run_video_script():
    """Trigger standard video script workflow (8-minute scripts from cards 01-10)"""
    subprocess.Popen(["python", "video_script_automation.py"])
    return "Video script workflow started", 200

@app.route('/run_video_script_short', methods=['POST'])
def run_video_script_short():
    """Trigger SHORT video script workflow (5-minute scripts from cards 11-15)"""
    subprocess.Popen(["python", "video_script_short_automation.py"])
    return "Short video script workflow started", 200

@app.route('/run_threads_titles', methods=['POST'])
def run_threads_titles():
    """Trigger threads & titles workflow (Twitter threads + video titles from cards 01-15)"""
    subprocess.Popen(["python", "threads_titles_automation.py"])
    return "Threads and titles workflow started", 200

@app.route('/run_email_sequence', methods=['POST'])
def run_email_sequence():
    """Trigger email sequence workflow (9-email sequences from cards 01-15)"""
    subprocess.Popen(["python", "email_sequence_automation.py"])
    return "Email sequence workflow started", 200

@app.route('/run_heygen_chunker', methods=['POST'])
def run_heygen_chunker():
    """Trigger HeyGen 4-problem script chunker workflow"""
    subprocess.Popen(["python", "heygen_4p_script_chunker_automation.py"])
    return "HeyGen 4-problem script chunker workflow started", 200

@app.route('/run_2p_heygen_chunker', methods=['POST'])
def run_2p_heygen_chunker():
    """Trigger HeyGen 2-problem script chunker workflow"""
    subprocess.Popen(["python", "heygen_2p_script_chunker_automation.py"])
    return "HeyGen 2-problem script chunker workflow started", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
    