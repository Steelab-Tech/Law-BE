import json
import pandas as pd
import ast
import openai
import os
import tqdm
import openai
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())  # Read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

def gen_answer(query, document):
    prompt = f"""
    As an intelligent virtual assistant, answer the question based on the provided documents. The answer should be concise, accurate, and cover all key points.
    # Question:
    {query}

    # Documents:
    {document}
    """
    response = openai.chat.completions.create(
        model= 'gpt-4o-mini',
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0,
    )

    return response.choices[0].message.content


train_data = "train.csv"
output_file = "gen_data/train.csv"

# Read data from train.csv
df = pd.read_csv(train_data)

# If output file exists, read it to continue from previous progress
if os.path.exists(output_file):
    df_output = pd.read_csv(output_file)
else:
    df_output = pd.DataFrame(columns=["question", "context", "answer"])

# Iterate through each row in DataFrame
for index, row in tqdm.tqdm(df.iterrows()):
    # Check if question has already been processed
    if 10001 <= index and index <= 11000:
        question = row['question']
        contexts = ast.literal_eval(row['context'])
        full_context = "Below is the complete document information: \n"
        for context in contexts:
            full_context += f"{context} \n"
        full_context += "End of document information."

        try:
            # Generate answer
            answer = gen_answer(question, full_context)

            # Add result to output DataFrame
            new_row = {"question": question, "context": full_context, "answer": answer}
            df_output = pd.concat([df_output, pd.DataFrame([new_row])], ignore_index=True)

            # Save immediately to CSV
            df_output.to_csv(output_file, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    else:
        continue
