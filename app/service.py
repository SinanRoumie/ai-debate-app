from agents import debater_a, debater_a_research_agent, debater_n_research_agent, debater_n, judge, message_storage, analyst
import re
import os
import json

rounds = [
    {"speaker": "Debater A", "agent": debater_a, "type": "constructive"},
    {"speaker": "Debater N", "agent": debater_n, "type": "constructive"},
    {"speaker": "Debater A", "agent": debater_a, "type": "rebuttal"},
    {"speaker": "Debater N", "agent": debater_n, "type": "rebuttal"},
    {"speaker": "Debater A", "agent": debater_a, "type": "summary"},
    {"speaker": "Debater N", "agent": debater_n, "type": "summary"},
    {"speaker": "Debater A", "agent": debater_a, "type": "final focus"},
    {"speaker": "Debater N", "agent": debater_n, "type": "final focus"}
]

def format_history(storage):
    return "\n\n".join(
        [f"{m['agent_type']} ({m['speech_type']}): {m['message']}" for m in storage]
    )

def save_analysis(result):
    try:
        with open("analyst_scores.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.append(result)

    with open("analyst_scores.json", "w") as f:
        json.dump(data, f, indent=2)

MAX_FEEDBACK_HISTORY = 2
FEEDBACK_FILE = "feedback_log.json"

# Load feedback from file if exists
if os.path.exists(FEEDBACK_FILE):
    try:
        with open(FEEDBACK_FILE, "r") as f:
            feedback_memory = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        feedback_memory = {
            "Debater A": [],
            "Debater N": []
    }
else:
    feedback_memory = {
        "Debater A": [],
        "Debater N": []
    }

def get_feedback_text(speaker):
        feedback_list = feedback_memory.get(speaker, [])
        if feedback_list:
            return "\nHere is feedback from the judge on your previous rounds:\n" + "\n".join([f"- {entry}" for entry in feedback_list]) + "\n"
        return ""

async def run_structured_debate(topic: str):
    global feedback_memory 
    message_storage.clear()

    # Run research once
    research_a = await debater_a_research_agent.run(
    f"Please use your search tool to find 10 relevant facts affirming the following debate topic: '{topic}'", 
    deps=topic  
)   

    research_n = await debater_n_research_agent.run(
    f"Please use your search tool to find 10 relevant facts negating the following debate topic: '{topic}'", 
    deps=topic
)

    # --- Constructive Speeches (manually written with research) ---
    constructive_a_prompt = (
        f"You are Debater A. You are in the constructive round of a structured debate.\n"
        f"Topic: {topic}\n"
        f"Please keep your responses under 400 words.\n"
        f"Use the following research to support your case:\n{research_a.output}\n\n"
        "Now write your argument:"
    )
    constructive_a = await debater_a.run(constructive_a_prompt)
    message_storage.append({
        "agent_type": "Debater A",
        "message": constructive_a.data,
        "speech_type": "constructive",
        "input_prompt": constructive_a_prompt
    })

    constructive_n_prompt = (
        f"You are Debater N. You are in the constructive round of a structured debate.\n"
        f"Topic: {topic}\n"
        f"Please keep your responses under 400 words.\n"
        f"Use the following research to support your case:\n{research_n.output}\n\n"
        "Now write your argument:"
    )
    constructive_n = await debater_n.run(constructive_n_prompt)
    message_storage.append({
        "agent_type": "Debater N",
        "message": constructive_n.data,
        "speech_type": "constructive",
        "input_prompt": constructive_n_prompt
    })

    # Map research for future use (if you want to still show it)
    research_data = {
        "Debater A": research_a.output,
        "Debater N": research_n.output
    }

    # --- Remaining Rounds (skip constructives since already done) ---
    for round_info in rounds:
        if round_info["type"] == "constructive":
            continue  # already handled above

        speaker = round_info["speaker"]
        agent = round_info["agent"]
        speech_type = round_info["type"]

        rules = ""
        if speech_type == "rebuttal":
            rules = "\nPlease keep your responses under 400 words."
        elif speech_type == "summary":
            rules = "\nPlease keep your responses under 300 words. Do not introduce any new evidence, only reference evidence previously mentioned."
        elif speech_type == "final focus":
            rules = "\nPlease keep your responses under 200 words. Do not introduce any new evidence, only reference evidence previously mentioned."

        history = format_history(message_storage)
        research = research_data.get(speaker, "")
        feedback_list = feedback_memory.get(speaker, [])
        if feedback_list:
            formatted_feedback = "\nHere is feedback from the judge on your previous rounds:\n"
            formatted_feedback += "\n".join([f"- {f}" for f in feedback_list]) + "\n"
        else:
            formatted_feedback = ""



        input_prompt = (
            f"You are {speaker}. You are in the {speech_type} round of a structured debate.\n"
            f"Topic: {topic}\n"
            f"{rules}"
            "Remember to respond to your opponent's arguments and defend your own"
            f"{formatted_feedback}\n"
            f"\nUse the following research to support your case:\n{research}\n\n"
            f"Here is the debate so far:\n{history}\n"
            "Now write your argument:"
        )


        result = await agent.run(input_prompt)
        message_storage.append({
            "agent_type": speaker,
            "speech_type": speech_type,
            "message": result.data,
            "input_prompt": input_prompt
        })

    # --- Judging Phase (unchanged) ---
    full_transcript = format_history(message_storage)
    judge_prompt = (
        f"Here is the full debate transcript:\n{full_transcript}\n\n"
        "Please decide who won the debate and explain your reasoning.\n"
        "Also, provide concise but specific feedback for each debater on how they can improve their arguments in future rounds.\n"
        "Format your response like:\n"
        "Winner: Debater A\n"
        "Feedback for Debater A: ...\n"
        "Feedback for Debater N: ..."
    )
    judgment = await judge.run(judge_prompt)
    
    message_storage.append({"agent_type": "Judge", "message": judgment.data})

    # Store feedback
    match_a = re.search(r"Feedback for Debater A:(.*?)(Feedback for Debater N:|$)", judgment.data, re.DOTALL)
    if match_a:
        feedback_a = match_a.group(1).strip()
        feedback_memory.setdefault("Debater A", []).append(feedback_a)
        feedback_memory["Debater A"] = feedback_memory["Debater A"][-MAX_FEEDBACK_HISTORY:]

    match_n = re.search(r"Feedback for Debater N:(.*)", judgment.data, re.DOTALL)
    if match_n:
        feedback_n = match_n.group(1).strip()
        feedback_memory.setdefault("Debater N", []).append(feedback_n)
        feedback_memory["Debater N"] = feedback_memory["Debater N"][-MAX_FEEDBACK_HISTORY:]

    # Save logs
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_memory, f, indent=2)

    with open("debate_log.json", "w") as f:
        json.dump(message_storage, f, indent=2)

    # Track wins
    winner = None
    for line in judgment.data.splitlines():
        if line.lower().startswith("winner:"):
            winner = line.split(":")[1].strip()
            break

    try:
        with open("results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    if winner:
        results[winner] = results.get(winner, 0) + 1

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    ## Analyst Run HERE ##
    

    # Step 1: Tagging Phase
    tagging_prompt = f"Here is the transcript of the round:\n\n{full_transcript}\n\nTag and extract as instructed."
    tagging_result = await analyst.run(tagging_prompt)

    try:
        observations = json.loads(tagging_result.output)
    except json.JSONDecodeError:
        print("❗ Failed to parse tagging output:")
        print(tagging_result.output)
        return None


    scoring_prompt = f"""
    You are AnalystBot. You have already extracted structured observations from a debate round.

    Here are the tagged observations:

    {json.dumps(observations, indent=2)}

    Now, using only this information, evaluate the technical debate quality of each side (Aff and Neg) based on the following metrics. Score each on a scale from 1 to 10:

1. ULI Adherence
2. Response Coverage
3. Depth of Clash
4. Argument Preservation

Return your results in this JSON format:

{{
"aff_scores": {{
    "uli_adherence": [1–10],
    "response_coverage": [1–10],
    "depth_of_clash": [1–10],
    "argument_preservation": [1–10],
    "average": [rounded average]
}},
"neg_scores": {{
    "uli_adherence": [1–10],
    "response_coverage": [1–10],
    "depth_of_clash": [1–10],
    "argument_preservation": [1–10],
    "average": [rounded average]
}}
}}
"""
    scoring_result = await analyst.run(scoring_prompt)
    
    try:
        scores = json.loads(scoring_result.output)
        save_analysis(scores)  # ✅ Now it's defined
        return scores
    except json.JSONDecodeError:
        print("❗ Failed to parse scoring output:")
        print(scoring_result.output)
        return None

    
    




