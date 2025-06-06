from flask import Flask, render_template_string
import json
import pandas as pd  # For reading .xlsx files

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

    try:
        with open("analyst_observations.json") as f:
            observations_data = json.load(f)
            latest_observations = observations_data[-1] if observations_data else None
    except FileNotFoundError:
        latest_observations = None

    try:
        aff_flow = pd.read_excel("aff_flow.xlsx").to_dict(orient="records")
    except FileNotFoundError:
        aff_flow = []

    try:
        neg_flow = pd.read_excel("neg_flow.xlsx").to_dict(orient="records")
    except FileNotFoundError:
        neg_flow = []

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
          .response {
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
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-x: auto;
          }
          .toggle-button {
            background-color: #007bff;
            color: white;
            padding: 6px 12px;
            border: none;
            cursor: pointer;
            font-size: 14px;
            border-radius: 5px;
            margin-top: 10px;
          }
          .hidden {
            display: none;
          }
        </style>
        <script>
          function togglePrompt(id) {
            const promptDiv = document.getElementById(id);
            promptDiv.classList.toggle('hidden');
          }
        </script>
      </head>
      <body>
        <h1>Debate Transcript</h1>

        {% if analysis %}
          <div class="message">
            <h2>📊 AnalystBot Scoring (Round {{ analysis.round }})</h2>
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

        {% if observations %}
          <div class="message">
            <h2>🧠 AnalystBot Tagging Observations (Round {{ observations.round }})</h2>
            <div class="response">
              <span class="label">ULI Arguments:</span>
              <pre>{{ observations.uli_arguments | tojson(indent=2) }}</pre>
            </div>
            <div class="response">
              <span class="label">Response Coverage:</span>
              <pre>{{ observations.response_coverage | tojson(indent=2) }}</pre>
            </div>
            <div class="response">
              <span class="label">Clash Chains:</span>
              <pre>{{ observations.clash_chains | tojson(indent=2) }}</pre>
            </div>
            <div class="response">
              <span class="label">Preservation:</span>
              <pre>{{ observations.preservation | tojson(indent=2) }}</pre>
            </div>
          </div>
        {% endif %}

        {% for entry in data %}
          <div class="message">
            <h2>{{ entry.agent_type }} - {{ entry.speech_type }}</h2>

            {% if entry.input_prompt %}
              <button class="toggle-button" onclick="togglePrompt('prompt-{{ loop.index }}')">Toggle Input Prompt</button>
              <div id="prompt-{{ loop.index }}" class="response hidden">
                <span class="label">Input Prompt:</span>
                <pre>{{ entry.input_prompt }}</pre>
              </div>
            {% endif %}

            <div class="response">
              <span class="label">Response:</span>
              {{ entry.message }}
            </div>
          </div>
        {% endfor %}

        {% if aff_flow %}
          <div class="message">
            <h2>🟦 AFF Flow Table</h2>
            {% for row in aff_flow %}
              <div class="response">
                <span class="label">{{ loop.index }}. {{ row.tag }}</span>
                {% for col, val in row.items() if col != 'tag' %}
                  <strong>{{ col.replace('_', ' ').title() }}:</strong> {{ val }}<br>
                {% endfor %}
              </div>
            {% endfor %}
          </div>
        {% endif %}

        {% if neg_flow %}
          <div class="message">
            <h2>🟥 NEG Flow Table</h2>
            {% for row in neg_flow %}
              <div class="response">
                <span class="label">{{ loop.index }}. {{ row.tag }}</span>
                {% for col, val in row.items() if col != 'tag' %}
                  <strong>{{ col.replace('_', ' ').title() }}:</strong> {{ val }}<br>
                {% endfor %}
              </div>
            {% endfor %}
          </div>
        {% endif %}
      </body>
    </html>
    """

    return render_template_string(
        template,
        data=log,
        analysis=latest_analysis,
        observations=latest_observations,
        aff_flow=aff_flow,
        neg_flow=neg_flow
    )

if __name__ == "__main__":
    app.run(debug=True)
