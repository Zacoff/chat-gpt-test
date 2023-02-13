import os

import openai
from flask import Flask, redirect, render_template, request, url_for
import redis
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

redis_conn = redis.Redis(host='localhost', port=6379, db=0)

document = pd.read_csv('test.csv')

len_def = document.shape[0]

title_vector = []
summary_vector = []
response_context = ''
best_context_info = []

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
            question = request.form["question"]
            for index in range(len_def):
                title = document.at[index, 'Title']
                context_to_ec_csv = document.at[index, 'Summary']
                summary_vector.append(context_to_ec_csv)
                title_vector.append(title)

            response_api = openai.Embedding.create(
                input=[
                    question,
                    *summary_vector
                ],
                model="text-embedding-ada-002"
            )

            question_embedding = response_api['data'][0]['embedding']
            
            for index in range(len_def):
                summary_embedding = response_api['data'][index + 1]['embedding']
                distance = np.dot(question_embedding, summary_embedding)
                line = index

                print(distance)

                if(len(best_context_info) == 0):
                    best_context_info.append(title_vector[index])
                    best_context_info.append(distance)
                    best_context_info.append(line)
                elif(best_context_info[1] < distance):
                    best_context_info[0] = title_vector[index]
                    best_context_info[1] = distance
                    best_context_info[2] = line
            
            response_context = document.at[best_context_info[2], 'Response']

            if(redis_conn.get(response_context[2]) == None):
                response_api = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=f'Essa é uma conversa acolhedora onde o cliente quer saber sobre {best_context_info[0]}, explique que {response_context}. Pergunte se pode ajudar em algo mais.\n\nQ:{question}\n\nA:',
                    max_tokens=400,
                    n=1,
                    stop='\n\n',
                    temperature=0.0,
                )

                response = response_api["choices"][0]["text"]

                test = redis_conn.get(response_context[2])

                redis_conn.set(response_context[2], f"{response}")
                redis_conn.expire(response_context[2], 60)
            
                return redirect(url_for("index", result=response))
            else:
                response = redis_conn.get(response_context[2]).decode("utf-8")
                return redirect(url_for("index", result=response))
            
    result = request.args.get("result")
    return render_template("index.html", result=result)