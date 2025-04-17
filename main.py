from flask import Flask
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return "Agentic Workflow Trigger is live!"

@app.route('/run_blog', methods=['POST'])
def run_blog():
    subprocess.Popen(["python", "combined_blog_workflow.py"])
    return "Blog workflow started", 200

@app.route('/run_reviews', methods=['POST'])
def run_reviews():
    subprocess.Popen(["python", "combined_reviews_workflow.py"])
    return "Reviews workflow started", 200

@app.route('/run_press_release', methods=['POST'])
def run_press_release():
    subprocess.Popen(["python", "combined_press_release_workflow.py"])
    return "Press release workflow started", 200

@app.route('/run_case_study', methods=['POST'])
def run_case_study():
    subprocess.Popen(["python", "combined_case_study_workflow.py"])
    return "Case study workflow started", 200

@app.route('/run_video_transcript', methods=['POST'])
def run_video_transcript():
    subprocess.Popen(["python", "combined_video_transcript_workflow.py"])
    return "Video transcript workflow started", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
