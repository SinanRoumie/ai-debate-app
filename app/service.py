from agents import debater_a, debater_a_research_agent, debater_n_research_agent, debater_n, judge, message_storage, analyst, flow
import re
import os
import json
import csv
import pandas as pd

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

def save_observations(observations):
    filename = "analyst_observations.json"
    try:
        with open(filename, "r") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        all_data = []

    all_data.append(observations)

    with open(filename, "w") as f:
        json.dump(all_data, f, indent=2)

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



def get_flow_instructions(speech_type, side, current_flow=""):
    col_map = {
        "constructive": "constructive",
        "rebuttal": "rebuttal",
        "summary": "summary",
        "final focus": "final_focus"
    }

    column = f"{side}_{col_map[speech_type]}"

    if speech_type == "constructive":
        return (
            f"In this {side.upper()} constructive speech, extract each argument as a new row. "
            "For each row, include:\n"
            f"- tag: a short label for the argument\n"
            f"- {column}: combine uniqueness (UQ): describe the status quo that sets up the argument, link (L): what action leads to the impact, and impact (!): the consequence or outcome\n\n"
            "Leave all other columns empty. Preserve all previous rows."
        )

    # For non-constructive speeches, include current flow table as context
    return (
        f"The current flow table is below:\n\n{current_flow}\n\n"
        f"In this {side.upper()} {speech_type} speech, update the column `{column}` in the {side.upper()} flow table.\n"
        "For each argument (matched by tag), either:\n"
        "- Extend it (summarize the extension)\n"
        "- Respond to it (briefly summarize)\n"
        "- Note if it was dropped\n\n"
        "Do not modify other columns or other table. Keep all previous content intact."
    )


def load_csv_as_dicts(path):
    try:
        with open(path, newline='') as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []

def save_dicts_to_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


## Run Structured Debate ##

async def run_structured_debate(topic: str):
    global feedback_memory 
    
    aff_flow = load_csv_as_dicts("aff_flow.csv")
    neg_flow = load_csv_as_dicts("neg_flow.csv")


    message_storage.clear()

    # Run research once
    research_a = await debater_a_research_agent.run(
    f"Find 10 pieces of evidence affirming the following topic: '{topic}'", 
    deps=topic  
    )   

    research_n = await debater_n_research_agent.run(
    f"Find 10 pieces of evidence negating the following topic: '{topic}'", 
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

    flow_input = f"""
    Instructions:
    {get_flow_instructions("constructive", "aff")}

    AFF Flow:
    {json.dumps(aff_flow, indent=2)}

    NEG Flow:
    {json.dumps(neg_flow, indent=2)}

    Speech:
    \"\"\"{constructive_a.data}\"\"\"
    """

    flow_result = await flow.run(flow_input)
    
    raw = flow_result.output.strip()

    # If GPT wrapped it in markdown-style triple backticks
    if raw.startswith("```"):
        raw = raw.strip("`")  # Remove all backticks
        parts = raw.split("json")
        raw = parts[-1].strip() if len(parts) > 1 else raw

    # Fix common JSON issues:
    # - Convert keys like aff_flow to "aff_flow"
    raw = re.sub(r'(?<=[^\\])",\s*"([L!])":', r', "\1":', raw)

    try:
        tables = json.loads(raw)
    except json.JSONDecodeError as e:
        print("❗ Failed to parse flow output:")
        print(raw)
        raise e
    
    tables = json.loads(flow_result.output)
    aff_flow = tables.get("aff_flow", aff_flow)
    neg_flow = tables.get("neg_flow", neg_flow)

    print("🔁 Updated AFF Flow:")
    print(json.dumps(aff_flow, indent=2))

    print("🔁 Updated NEG Flow:")
    print(json.dumps(neg_flow, indent=2))




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

    flow_input = f"""
    Instructions:
    {get_flow_instructions("constructive", "neg")}

    AFF Flow:
    {json.dumps(aff_flow, indent=2)}

    NEG Flow:
    {json.dumps(neg_flow, indent=2)}

    Speech:
    \"\"\"{constructive_n.data}\"\"\"
    """

    flow_result = await flow.run(flow_input)
    print("FLOWBOT OUTPUT:")
    print(flow_result.output)
    tables = json.loads(flow_result.output)
    aff_flow = tables.get("aff_flow", aff_flow)
    neg_flow = tables.get("neg_flow", neg_flow)




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

        side = "aff" if speaker == "Debater A" else "neg"

        flow_input = f"""
        Instructions:
        {get_flow_instructions(speech_type, side, current_flow=json.dumps(aff_flow if side == "aff" else neg_flow, indent=2))}

        AFF Flow:
        {json.dumps(aff_flow, indent=2)}

        NEG Flow:
        {json.dumps(neg_flow, indent=2)}

        Speech:
        \"\"\"{result.data}\"\"\"
        """

        flow_result = await flow.run(flow_input)
        tables = json.loads(flow_result.output)

        # Normalize keys from FlowBot output
        aff_flow = tables.get("AFF Flow", aff_flow)
        neg_flow = tables.get("NEG Flow", neg_flow)

        print("🔁 Updated AFF Flow:")
        print(json.dumps(aff_flow, indent=2))

        print("🔁 Updated NEG Flow:")
        print(json.dumps(neg_flow, indent=2))


        message_storage.append({
            "agent_type": speaker,
            "speech_type": speech_type,
            "message": result.data,
            "input_prompt": input_prompt
        })

    
    save_dicts_to_csv("aff_flow.csv", aff_flow)
    save_dicts_to_csv("neg_flow.csv", neg_flow)
    
    pd.DataFrame(aff_flow).to_excel("aff_flow.xlsx", index=False)
    pd.DataFrame(neg_flow).to_excel("neg_flow.xlsx", index=False)



    # --- Judging Phase (unchanged) ---
    full_transcript = format_history(message_storage)
    judge_prompt = (
        f"Here is the full debate transcript:\n{full_transcript}\n\n"
        "Please decide who won the debate and explain which argument(s) won them the debate and why.\n"
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
    

#     # Step 1: Tagging Phase
#     tagging_prompt = f"Here is the transcript of the round:\n\n{full_transcript}\n\nTag and extract as instructed."
#     tagging_result = await analyst.run(tagging_prompt)

#     try:
#         raw = tagging_result.output.strip()
#         if raw.startswith("```"):
#             raw = raw.strip("`")  # Strip all backticks
#             parts = raw.split("json")
#             raw = parts[-1].strip() if len(parts) > 1 else raw

#         observations = json.loads(raw)
#         save_observations(observations)
#     except json.JSONDecodeError:
#         print("❗ Failed to parse tagging output:")
#         print(tagging_result.output)
#         return None


#     scoring_prompt = f"""
#     You are AnalystBot. You have already extracted structured observations from a debate round.

#     Here are the tagged observations:

#     {json.dumps(observations, indent=2)}

#     Now, using only this information, evaluate the technical debate quality of each side (Aff and Neg) based on the following metrics. Score each on a scale from 1 to 10:

# 1. ULI Adherence
# 2. Response Coverage
# 3. Depth of Clash
# 4. Argument Preservation

# Return your results in this JSON format:

# {{
# "aff_scores": {{
#     "uli_adherence": [1–10],
#     "response_coverage": [1–10],
#     "depth_of_clash": [1–10],
#     "argument_preservation": [1–10],
#     "average": [rounded average]
# }},
# "neg_scores": {{
#     "uli_adherence": [1–10],
#     "response_coverage": [1–10],
#     "depth_of_clash": [1–10],
#     "argument_preservation": [1–10],
#     "average": [rounded average]
# }}
# }}
# """
#     scoring_result = await analyst.run(scoring_prompt)

#     try:
#         scores = json.loads(scoring_result.output)
#         save_analysis(scores)  # ✅ Now it's defined
#         return scores
#     except json.JSONDecodeError:
#         print("❗ Failed to parse scoring output:")
#         print(scoring_result.output)
#         return None

    
    




