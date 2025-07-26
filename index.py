
import json
import spacy
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import unicodedata

# Descargar recursos de NLTK
try:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    print("Error descargando recursos NLTK")

# Cargar el modelo spaCy en español
try:
    nlp = spacy.load("es_core_news_md", disable=["parser", "ner"])
except:
    print("Error: Instala el modelo de spaCy: python -m spacy download es_core_news_md")
    nlp = None

class ChatBot:
    def __init__(self, json_path="answers_FAQ.json"):
        self.faqs = self.cargar_faqs(json_path)
        self.conectores = {"y", "o", "pero", "porque", "sin", "embargo", "aunque", "además", 
                          "por", "lo", "tanto", "así", "que", "eso", "entonces", "ni", 
                          "cuando", "donde", "como", "para", "con", "de", "la", "el", "en"}
        
        # Cargar modelo de embeddings semánticos
        try:
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except:
            print("Error cargando sentence-transformers. Instala: pip install sentence-transformers")
            self.sentence_model = None
        
        self.preparar_datos()
        self.entrenar_modelos()
    
    def cargar_faqs(self, json_path):
        """Cargar y expandir el conjunto de FAQs"""

        with open(json_path, encoding="utf-8") as f:
            faqs = json.load(f)

        return faqs
    
    def normalizar_texto(self, texto):
        """Normalización avanzada del texto"""
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Remover acentos
        texto = unicodedata.normalize('NFD', texto)
        texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
        
        # Remover signos de puntuación y caracteres especiales, pero mantener espacios
        texto = re.sub(r'[^\w\s]', ' ', texto)
        
        # Remover espacios múltiples
        texto = re.sub(r'\s+', ' ', texto).strip()
        
        return texto
    
    def lematizar_y_expander_mejorado(self, texto):
        """Lematización mejorada con expansión de sinónimos y normalización"""
        if not nlp:
            return self.normalizar_texto(texto)
        
        texto_normalizado = self.normalizar_texto(texto)
        doc = nlp(texto_normalizado)
        lemmas = []
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                token.lemma_.strip() and 
                token.text not in self.conectores and
                len(token.text) > 2):  # Filtrar palabras muy cortas
                
                # Añadir el lema
                lemmas.append(token.lemma_)
                
                # Expandir con sinónimos (de forma controlada)
                try:
                    synsets = wn.synsets(token.text, lang='spa')
                    if synsets and len(lemmas) < 50:  # Limitar expansión
                        syn = synsets[0]
                        synonyms = [lemma.name().replace('_', ' ') for lemma in syn.lemmas('spa')]
                        if synonyms:
                            lemmas.append(synonyms[0])
                except:
                    pass
        
        return " ".join(lemmas)
    
    def generar_variaciones(self, pregunta):
        """Generar variaciones de una pregunta para augmentar datos"""
        variaciones = [pregunta]
        
        # Variación sin signos de interrogación
        sin_interrogacion = re.sub(r'[¿?]', '', pregunta).strip()
        if sin_interrogacion != pregunta:
            variaciones.append(sin_interrogacion)
        
        # Variación como afirmación
        if pregunta.startswith('¿Cómo'):
            afirmacion = pregunta.replace('¿Cómo', 'Necesito saber cómo').replace('?', '')
            variaciones.append(afirmacion)
        elif pregunta.startswith('¿Qué'):
            afirmacion = pregunta.replace('¿Qué', 'Quiero saber qué').replace('?', '')
            variaciones.append(afirmacion)
        elif pregunta.startswith('¿Puedo'):
            afirmacion = pregunta.replace('¿Puedo', 'Quiero').replace('?', '')
            variaciones.append(afirmacion)
        
        # Variación informal
        informal = pregunta.replace('¿', '').replace('?', '').lower()
        if not informal.startswith(('como', 'que', 'cuando', 'donde')):
            informal = f"me puedes decir {informal}"
        variaciones.append(informal)
        
        return variaciones
    
    def preparar_datos(self):
        """Preparar datos con augmentación y múltiples representaciones"""
        self.X_train_text = []
        self.X_train_lemma = []
        self.y_train = []
        self.embeddings_train = []
        
        # Crear mapeo de intents a respuestas
        self.intent_to_answer = {}
        for entry in self.faqs:
            self.intent_to_answer[entry["intent"]] = entry["answer"]
        
        # Preparar datos con augmentación
        for entry in self.faqs:
            intent = entry["intent"]
            for pregunta in entry["questions"]:
                # Generar variaciones de cada pregunta
                variaciones = self.generar_variaciones(pregunta)
                
                for variacion in variaciones:
                    # Texto original normalizado
                    texto_normalizado = self.normalizar_texto(variacion)
                    self.X_train_text.append(texto_normalizado)
                    
                    # Texto lematizado
                    texto_lematizado = self.lematizar_y_expander_mejorado(variacion)
                    self.X_train_lemma.append(texto_lematizado)
                    
                    self.y_train.append(intent)
        
        # Generar embeddings semánticos
        if self.sentence_model:
            self.embeddings_train = self.sentence_model.encode(self.X_train_text)
        
        print(f"Datos de entrenamiento preparados: {len(self.X_train_text)} ejemplos")
        print(f"Distribución de clases: {dict(zip(*np.unique(self.y_train, return_counts=True)))}")
    
    def entrenar_modelos(self):
        """Entrenar múltiples modelos y crear ensemble"""
        # Dividir datos
        indices = list(range(len(self.X_train_text)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, 
                                            stratify=self.y_train, random_state=42)
        
        X_train_text = [self.X_train_text[i] for i in train_idx]
        X_val_text = [self.X_train_text[i] for i in val_idx]
        X_train_lemma = [self.X_train_lemma[i] for i in train_idx]
        X_val_lemma = [self.X_train_lemma[i] for i in val_idx]
        y_train = [self.y_train[i] for i in train_idx]
        y_val = [self.y_train[i] for i in val_idx]
        
        # Modelo 1: TF-IDF con texto normalizado
        self.pipeline_text = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3), 
                max_features=5000,
                min_df=1,
                max_df=0.95
            )),
            ("clf", RandomForestClassifier(
                n_estimators=200, 
                random_state=42, 
                class_weight='balanced',
                max_depth=20
            ))
        ])
        
        # Modelo 2: TF-IDF con texto lematizado
        self.pipeline_lemma = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=3000
            )),
            ("clf", SVC(
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        # Entrenar modelos
        self.pipeline_text.fit(X_train_text, y_train)
        self.pipeline_lemma.fit(X_train_lemma, y_train)
        
        # Evaluación
        y_pred_text = self.pipeline_text.predict(X_val_text)
        y_pred_lemma = self.pipeline_lemma.predict(X_val_lemma)
        
        print(f"Accuracy modelo texto: {accuracy_score(y_val, y_pred_text):.4f}")
        print(f"Accuracy modelo lematizado: {accuracy_score(y_val, y_pred_lemma):.4f}")
        
        # Preparar embeddings para búsqueda semántica
        if self.sentence_model:
            self.embeddings_by_intent = {}
            for i, intent in enumerate(self.y_train):
                if intent not in self.embeddings_by_intent:
                    self.embeddings_by_intent[intent] = []
                self.embeddings_by_intent[intent].append(self.embeddings_train[i])
            
            # Promediar embeddings por intent
            for intent in self.embeddings_by_intent:
                self.embeddings_by_intent[intent] = np.mean(
                    self.embeddings_by_intent[intent], axis=0
                )
        
        print("Modelos entrenados exitosamente!")
    
    def busqueda_semantica(self, texto_usuario):
        """Búsqueda semántica usando embeddings"""
        if not self.sentence_model:
            return None, 0.0
        
        # Generar embedding del texto del usuario
        user_embedding = self.sentence_model.encode([texto_usuario])
        
        mejor_intent = None
        mejor_similitud = 0.0
        
        # Comparar con embeddings promedio de cada intent
        for intent, intent_embedding in self.embeddings_by_intent.items():
            similitud = cosine_similarity(user_embedding, 
                                        intent_embedding.reshape(1, -1))[0][0]
            if similitud > mejor_similitud:
                mejor_similitud = similitud
                mejor_intent = intent
        
        return mejor_intent, mejor_similitud
    
    def predecir_intencion(self, texto_usuario):
        """Predicción mejorada combinando múltiples enfoques"""
        if not texto_usuario.strip():
            return "Por favor, escribe tu pregunta para poder ayudarte."
        
        # Normalizar entrada del usuario
        texto_normalizado = self.normalizar_texto(texto_usuario)
        texto_lematizado = self.lematizar_y_expander_mejorado(texto_usuario)
        
        # Predicciones de modelos tradicionales
        pred_text = self.pipeline_text.predict([texto_normalizado])[0]
        proba_text = max(self.pipeline_text.predict_proba([texto_normalizado])[0])
        
        pred_lemma = self.pipeline_lemma.predict([texto_lematizado])[0]
        proba_lemma = max(self.pipeline_lemma.predict_proba([texto_lematizado])[0])
        
        # Búsqueda semántica
        pred_semantic, similitud_semantic = self.busqueda_semantica(texto_usuario)
        
        # Combinar predicciones (voting ponderado)
        predicciones = {}
        
        # Ponderar por confianza
        if pred_text in predicciones:
            predicciones[pred_text] += proba_text * 0.4
        else:
            predicciones[pred_text] = proba_text * 0.4
            
        if pred_lemma in predicciones:
            predicciones[pred_lemma] += proba_lemma * 0.3
        else:
            predicciones[pred_lemma] = proba_lemma * 0.3
        
        if pred_semantic and similitud_semantic > 0.3:
            if pred_semantic in predicciones:
                predicciones[pred_semantic] += similitud_semantic * 0.3
            else:
                predicciones[pred_semantic] = similitud_semantic * 0.3
        
        # Seleccionar la mejor predicción
        if predicciones:
            mejor_intent = max(predicciones, key=predicciones.get)
            confianza_final = predicciones[mejor_intent]
        else:
            mejor_intent = pred_text
            confianza_final = proba_text
        
        print(f"Predicción final: {mejor_intent} (confianza: {confianza_final:.4f})")
        print(f"Desglose - Texto: {pred_text}({proba_text:.3f}), "
              f"Lemma: {pred_lemma}({proba_lemma:.3f}), "
              f"Semántico: {pred_semantic}({similitud_semantic:.3f})")
        
        # Umbral de confianza adaptativo
        umbral_base = 0.25
        if confianza_final >= umbral_base:
            return self.intent_to_answer.get(mejor_intent, 
                "Lo siento, no tengo información específica sobre esa consulta.")
        else:
            return ("Lo siento, no estoy seguro de entender tu pregunta. "
                   "¿Podrías reformularla de otra manera? Puedo ayudarte con temas sobre "
                   "registro de clientes, automatización, hardware, software, casos de éxito, "
                   "tarjetas NFC y ciberseguridad de INGE-LEAN.")

# # Uso del chatbot mejorado
# if __name__ == "__main__":
#     # Crear e inicializar el chatbot
#     print("Inicializando chatbot mejorado...")
#     chatbot = ChatbotMejorado()
    
#     # Ejemplos de uso
#     ejemplos = [
#        ''' "¿Qué es la interconexión digital?",
#         "Que es la interconexión digital",  # Sin signos de interrogación
#         "Necesito saber sobre interconexión digital",  # Parafraseo
#         "me puedes explicar la interconexión digital",  # Informal
#         "Como me registro",  # Sin signos
#         "Quiero registrarme en la plataforma",  # Parafraseo
#         "Información sobre automatización de procesos",'''
#          "Cómo integrar la automatización con el hardware"
#     ]
    
#     print("\n" + "="*50)
#     print("PROBANDO EL CHATBOT MEJORADO")
#     print("="*50)
    
#     for ejemplo in ejemplos:
#         print(f"\nUsuario: {ejemplo}")
#         respuesta = chatbot.predecir_intencion(ejemplo)
#         print(f"Bot: {respuesta}")
#         print("-" * 40)
