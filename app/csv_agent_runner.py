# csv_agent_runner.py
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI

import os

import pandas as pd
from io import StringIO

def run_csv_agent(csv_file_path, prompt):
    agent = create_csv_agent(
        ChatOpenAI(model="gpt-4", temperature=0),
        csv_file_path,
        verbose=True
    )

    response = agent.run(prompt)

    # If response is a CSV string, save it
    try:
        df = pd.read_csv(StringIO(response))
        df.to_csv(csv_file_path, index=False)
        print(f"✅ CSV saved to {csv_file_path}")
    except Exception as e:
        print("⚠️ Couldn't parse agent output as CSV:", e)

    return response
