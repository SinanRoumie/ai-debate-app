from flask import Flask, render_template_string
import json

app = Flask(__name__)

@app.route("/")
def view_debate():
    try:
        with open("debate_log.json") as f:
            log = json.load(f)
    except FileNotFoundError:
        log = []

    try:
        with open("analyst_scores.json") as f:
            analysis_data = json.load(f)
            latest_analysis = analysis_data[-1] if analysis_data else None
    except FileNotFoundError:
        latest_analysis = None

    template = """
    <html>
      <head>
        <title>Debate Viewer</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: auto;
            background-color: #f5f5f5;
            padding: 20px;
          }
          .message {
            background-color: #fff;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 25px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
          }
          h2 {
            margin-top: 0;
          }
          .prompt, .response {
            white-space: pre-wrap;
            word-wrap: break-word;
            margin-top: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 4px solid #007BFF;
          }
          .label {
            font-weight: bold;
            margin-bottom: 5px;
            display: block;
          }
          pre {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
          }
        </style>
      </head>
      <body>
        <h1>Debate Transcript with Inputs</h1>

        {% if analysis %}
          <div class="message">
            <h2>📊 AnalystBot Evaluation (Round {{ analysis.round }})</h2>
            <div class="response">
              <span class="label">Aff Scores:</span>
              <pre>{{ analysis.aff_scores | tojson(indent=2) }}</pre>
            </div>
            <div class="response">
              <span class="label">Neg Scores:</span>
              <pre>{{ analysis.neg_scores | tojson(indent=2) }}</pre>
            </div>
          </div>
        {% endif %}

        {% for entry in data %}
          <div class="message">
            <h2>{{ entry.agent_type }} - {{ entry.speech_type }}</h2>
            {% if entry.input_prompt %}
              <div class="prompt">
                <span class="label">Prompt:</span>
                {{ entry.input_prompt }}
              </div>
            {% endif %}
            <div class="response">
              <span class="label">Response:</span>
              {{ entry.message }}
            </div>
          </div>
        {% endfor %}
      </body>
    </html>
    """
    return render_template_string(template, data=log, analysis=latest_analysis)

if __name__ == "__main__":
    app.run(debug=True)
