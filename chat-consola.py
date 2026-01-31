import nltk
import random
import string
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

#Configuracion inicial

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Descargas necesarias (solo se ejecutan si no existen)
# nltk.download('punkt', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt_tab', quiet=True)

#CARGA DEL CORPUS
try:
    with open(r'C:\Python\chatbot\Corpus_Encarnacion.txt', 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read().lower()
except FileNotFoundError:
    print("Error: No se encontró el archivo 'Corpus_Encarnacion.txt'. Revisa la ruta.")
    raw = ""

#PREPROCESAMIENTO
sent_tokens = nltk.sent_tokenize(raw)
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#FUNCIÓN DE RESPUESTA
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    
    
    TfidfVec = TfidfVectorizer(
        tokenizer=LemNormalize, 
        stop_words=stopwords.words('spanish'),
        token_pattern=None  # <--- ESTO ELIMINA EL ERROR
    )
    
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf == 0):
        robo_response = "Lo siento, no te he entendido. Si no puedo responder a lo que busca póngase en contacto con soporte@soporte.com"
    else:
        robo_response = sent_tokens[idx]
    
    return robo_response

# 4. COINCIDENCIAS MANUALES
SALUDOS_INPUTS = ("hola", "buenas", "saludos", "qué+e tal", "hey", "buenos dias")
SALUDOS_OUTPUTS = ["Hola", "Hola, ¿Que tal?", "Hola, ¿Cómo te puedo ayudar?", "Hola, encantado de hablar contigo"]

def saludos(sentence):
    for word in sentence.split():
        if word.lower() in SALUDOS_INPUTS:
            return random.choice(SALUDOS_OUTPUTS)

# 5. BUCLE PRINCIPAL
print("ROBOT: Mi nombre es ROBOT. Contestaré a tus preguntas acerca de Encarnación. Escribe 'salir' para terminar.")

while True:
    user_response = input("Tú: ").lower()
    
    if user_response != 'salir':
        if user_response in ['gracias', 'muchas gracias']:
            print("ROBOT: No hay de qué")
            break
        else:
            if saludos(user_response) is not None:
                print("ROBOT: " + saludos(user_response))
            else:
                print("ROBOT: ", end="")
                res = response(user_response)
                print(res)
                sent_tokens.remove(user_response)
    else:
        print("ROBOT: Nos vemos pronto, ¡cuídate!")
        break