import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    r.raise_for_status()

    embedding = r.json()["embeddings"]
    return embedding


def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-r1:1.5b",
            "prompt": prompt,
            "stream": False
        }
    )

    r.raise_for_status()

    response = r.json()
    print(response)
    return response


df = joblib.load("embeddings.joblib")


incoming_query = input("Ask a Question: ")

question_embedding = create_embedding([incoming_query])[0]


# Find similarities of question_embedding with other embeddings
similarities = cosine_similarity(
    np.vstack(df["embedding"]),
    [question_embedding]
).flatten()


top_results = 5

max_indx = similarities.argsort()[::-1][0:top_results]

new_df = df.loc[max_indx]


prompt = f'''I am helping the user learn from an MIT OpenCourseWare video lecture course.

Here are the most relevant video subtitle chunks retrieved from the course. Each chunk contains the video title, lecture number, start time in seconds, end time in seconds, and the spoken text:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}

---------------------------------

"{incoming_query}"

Answer the user's question based ONLY on the provided MIT OpenCourseWare video chunks.

Answer in a clear, natural and helpful way.

If the question is related to the course content:
- Explain the answer using the retrieved lecture content.
- Clearly mention which lecture/video contains the relevant information.
- Mention the approximate timestamp where the relevant topic is discussed.
- If multiple lectures are relevant, mention each relevant lecture and timestamp.
- Guide the user to watch the relevant portion of the lecture.
- Do not mention that you are looking at "chunks", "embeddings", "JSON", or the retrieval process.

If the retrieved content does not contain enough information to answer the question, say that the available lecture content does not provide enough information rather than making up an answer.

If the user asks something unrelated to the MIT course, politely tell them that you can only answer questions related to the course.
'''


with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)


# Generate the final answer using DeepSeek-R1 1.5B
response = inference(prompt)["response"]

print(response)


with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)