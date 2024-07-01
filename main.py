from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import psycopg2
from constant import dbname, user, password, host, port, API_KEY
import math

client = MistralClient(api_key=API_KEY)
def dot_product(vector1, vector2):
    return sum(v1 * v2 for v1, v2 in zip(vector1, vector2))

def vector_norm(vector):
    return math.sqrt(sum(v ** 2 for v in vector))

def cosine_similarity(vector1, vector2):
    dot_prod = dot_product(vector1, vector2)
    norm1 = vector_norm(vector1)
    norm2 = vector_norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_prod / (norm1 * norm2)

def get_text_embedding(input):
    embeddings_batch_response = client.embeddings(
          model="mistral-embed",
          input=input
      )
    return embeddings_batch_response.data[0].embedding


conn_params = {
    'dbname': dbname,
    'user': user,
    'password': password,
    'host': host,
    'port': port
}
def run_mistral(user_message, model="mistral-medium-latest"):
    messages = [
        ChatMessage(role="user", content=user_message)
    ]
    chat_response = client.chat(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)



def connect_to_db(params):
    try:
        connection = psycopg2.connect(**params)
        return connection
    except Exception as e:
        print(f"Erreur lors de la connexion à la base de données : {e}")
        return None

question = input("Posez une question sur le stage : ")
question_embeddings = [get_text_embedding(question)]
print("Question ok")

connection = connect_to_db(conn_params)


if connection:
    print("Connexion ok")

    select_query = """
        SELECT vector_data, id, chunks FROM vectordata3;
        """
    with connection.cursor() as cursor:
        cursor.execute(select_query)
        vectors = cursor.fetchall()
        print("Vecteurs récupérés")
        print(len(vectors))

        best_id = 0
        initial_similarity = 0
        best_chunks = ""

        for vector in vectors:
            vector_data = vector[0]
            vector_data = vector_data[1:-1]
            vector_data = vector_data.split(',')
            vector_data = [float(i) for i in vector_data]

            id = vector[1]
            chunks = vector[2]
            similarity = cosine_similarity(vector_data, question_embeddings[0])
            print("id: ", id, " similarity : ", similarity, "chunks : ", chunks)
            if similarity > initial_similarity:
                initial_similarity = similarity
                best_id = id
                best_chunks = chunks
            initial_similarity = similarity

    connection.close()
    print("Connexion fermée")

print("best id : ", best_id)
print("best chunks : ", best_chunks)

prompt = f"""
Context information is below.
---------------------
{best_chunks}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""
run_mistral(prompt)


