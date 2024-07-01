from mistralai.client import MistralClient
import psycopg2
from constant import dbname, user, password, host, port, API_KEY
import unicodedata

client = MistralClient(api_key=API_KEY)

with open('text.txt', 'r', encoding='utf-8') as fichier:
    maxime_story = fichier.read()
def remove_accents(input_str):
    # Normalisation de la chaîne en forme NFD (Normalization Form Decomposition)
    nfkd_form = unicodedata.normalize('NFD', input_str)
    # Filtrer et conserver uniquement les caractères non combinés (non-accentués)
    only_ascii = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    return only_ascii

def get_text_embedding(input):
    embeddings_batch_response = client.embeddings(
          model="mistral-embed",
          input=input
      )
    return embeddings_batch_response.data[0].embedding

chunk_size = 1000
maxime_story = remove_accents(maxime_story)
chunks = [maxime_story[i:i + chunk_size] for i in range(0, len(maxime_story), chunk_size)]
len(chunks)

conn_params = {
    'dbname': dbname,
    'user': user,
    'password': password,
    'host': host,
    'port': port
}

def connect_to_db(params):
    try:
        connection = psycopg2.connect(**params)
        return connection
    except Exception as e:
        print(f"Erreur lors de la connexion à la base de données : {e}")
        return None

def create_table(connection):
    try:
        with connection.cursor() as cursor:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS vectordata3 (
                id SERIAL PRIMARY KEY,
                vector_data VECTOR(1024),
                chunks TEXT
            );
            """
            cursor.execute(create_table_query)
            connection.commit()
    except Exception as e:
        print(f"Erreur lors de la création de la table : {e}")


def insert_vector_data(connection, vector, chunks):
    try:
        with connection.cursor() as cursor:
            insert_query = """
            INSERT INTO vectordata3 (vector_data,chunks) VALUES (%s,%s);
            """
            cursor.execute(insert_query, (vector, chunks))
            connection.commit()
    except Exception as e:
        print(f"Erreur lors de l'insertion des données : {e}")

connection = connect_to_db(conn_params)

text_embeddings = [get_text_embedding(chunk) for chunk in chunks]

if connection:
    print("Connexion ok")

    create_table(connection)
    for i in range(len(text_embeddings)):
        insert_vector_data(connection, text_embeddings[i], chunks[i])

    print("Données insérées")
    connection.close()
    print("Connexion fermée")