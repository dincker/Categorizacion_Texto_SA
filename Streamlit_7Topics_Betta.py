#!/usr/bin/env python
# coding: utf-8

# In[1]:
print('Content-Type: text/plain')
print('')

#get_ipython().run_line_magic('pip', 'install streamlit')
#get_ipython().run_line_magic('pip', 'install pickle')
#get_ipython().run_line_magic('pip', 'install numpy')


# In[1]:


import streamlit as st
import pickle
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
nlp = spacy.load('es_core_news_md')
spanishstemmer = SnowballStemmer('spanish')


# In[2]:


vocabulary = pickle.load(open("vocabulary_7_V2.pk", 'rb'))


# In[51]:


#VECTORIZACION

def normalize(text):
  #Tokenizacion spacy
  doc = nlp(text)
  #Lemmatizacion | Eliminacion de caracteres |Eliminacion de palabras de parada
  words = [t.lemma_ for t in doc if not t.is_punct | t.is_stop]
  #Texto a minusculas | Eliminacion de caracteres alfanumericos
  lexical_tokens = [t.lower() for t in words if len(t) > 2 and t.isalpha()]
  #Transformacion de numero a letras - Ingles
  #lexical_tokens=replace_numbers(lexical_tokens)
  #Elimina palabras que no se encuentran en el codigo ASCII
  #lexical_tokens=remove_non_ascii(lexical_tokens)
  #Stemmer
  stema = [spanishstemmer.stem(token) for token in lexical_tokens]
  return stema
#import nltk
#spanish_stemmer = nltk.stem.SnowballStemmer('spanish')

#///////////////////////////////////////////////////
#VECTORIZACION

#LIBRERIAS


#Ajuste de parametros de vectorizacion
vectorizer_tfidf = TfidfVectorizer(
                      analyzer='word',
                      tokenizer= normalize,
                      lowercase= True,
                      vocabulary = vocabulary,
                      stop_words=nlp.Defaults.stop_words,
                      min_df=1, #Elimina palabras que aparecen menos de 3 veces
                      max_df=0.98) #Elimina palabras que aparecen en el 95% de los textos

#El ajuste, realiza la normalizacion, eliminacion de stop-words y sin relevancia, ademas de crea el vocabulario.(Aplica los parametros establecidos)


# In[96]:


bag_of_wors_idf = pickle.load(open("bag_of_words_7_topics.pk",'rb'))


# In[97]:


count_train_tfidf = pickle.load(open("tfidf_7_vector_topics.pk", 'rb'))


# In[98]:


best_lda_model = pickle.load(open("LDA_MODEL_7_TOPICS.pk", 'rb'))


# In[99]:


#best_lda = pickle.load(open("LDA_2_7.pk", 'rb'))


# In[100]:


#topic_models = pickle.load(open("topic_models.pk", 'rb'))


# In[101]:


#tfidf_7_vector_topics
#best_lda_model.transform


# In[102]:


#k= topic_models[0]
#W = topic_models[1]
#H = topic_models[2]
#best_lda_model.transform(vocabulary)


# In[103]:


#bag_of_wors_idf
#count_train_tfidf.transform(vocabulary)
#count_train_tfidf.transform


# In[104]:


#Tema dominante en el texto
k = best_lda_model.n_components
# obtener el modelo que generamos anteriormente.
#W = topic_models[5][1]
#H = topic_models[5][2] 
W = best_lda_model.transform(bag_of_wors_idf)#best_lda_model.transform(bag_of_wors_idf)
H = best_lda_model.components_


# In[105]:


#pickle.dump(model, open("LDA_MODEL_7_V1_TOPICS.pk", 'wb'))


# In[106]:



#Nombre de columnas
topicnames = []
for i in range(len(H)):#best_lda_model.n_components)):
    name='Topic'+str(i)
    topicnames.append(name)


# In[107]:


#W


# In[108]:


#nombre de indices
docnames = []
#for i in range(len(cv.get_feature_names())):
for i in range(len(W)):
    docnames.append('Doc'+str(i))


# In[109]:



df_document_topic=pd.DataFrame(np.round(W,4),columns=topicnames,index=docnames)
maximo = df_document_topic.max(axis=1)
id_maximo = df_document_topic.idxmax(axis=1)
df_document_topic['dominant_topic'] = id_maximo
df_document_topic['width_topic'] = maximo


# In[ ]:





# In[110]:


#count_train_tfidf.fit()


# In[111]:


#Obtenga las palabras clave de cada tema (Opcion)
# Topic - Keyword matrix
df_topic_keywords = pd.DataFrame(H)

# assign column and index
df_topic_keywords.columns = vectorizer_tfidf.get_feature_names()
df_topic_keywords.index = topicnames


# In[112]:


# Mostrat las top n palabras por topico
def show_topics(vectorizer=vectorizer_tfidf, component=H, n_words=10):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in component:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


# In[162]:


topic_keywords = show_topics(vectorizer_tfidf, H, 30)
#len(topic_keywords[0])


# In[163]:


# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]


# In[164]:


#ta = nlp("acceso")
#words = [t.lemma_ for t in ta if not t.is_punct | t.is_stop]
#lexical_tokens = [t.lower() for t in words if len(t) > 2 and t.isalpha()]
#stema = [spanishstemmer.stem(token) for token in lexical_tokens]
#stema


# In[165]:


#df_topic_keywords.to_excel("DS.xlsx")


# In[181]:


st.header("Ingrese un texto para categorizar")
#x="HOla, solicito un cambio de contrase??a"
x = st.text_input("Ingrese un texto a categorizar")


# In[174]:


Topic=["Requerimiento","Reparacion","Reinicio de clave","Retira","Cambio de Contrase??a","Vehiculo de transporte","indi","Transporte"]
#Topic = ["Desbloqueo","Error de ubicacion","Solicitud de acceso","Despacho de equipo"]
df_topic_keywords["Topic"]=Topic


# In[179]:


#st.header("Ingrese un texto para categorizar")
#x = st.text_input()
#if st.button("Calcula"):
    
#TRANSPORTE
#text="Estimados, Solicito a ustedes una camioneta entre las fechas indicadas en el formulario"#, con el motivo de asistir a una visita en terreno para el programa GROT en las plantas de Canela Alta, Canela Baja y los Vilos. Si hubiera la posibilidad de que esta fuera retirada en Vi??a del Mar les estar??a mas que agradecido. Gracias por adelantado. Saludos, Andres Cordova M"
#text="Se solicita veh??culo para visita a plantas Hijuelas"# - Centenario - Oriente - Pachacama (La Cruz - Hijuelas)# con motivo de revisi??n de sistemas de dosificaci??n de fl??or y levantamiento GROT en San Jer??nimo (San Antonio) Muchas gracias. Saludos.
#text="Estimados solicito trasladar camioneta KXXJ-26 desde recinto Uno Norte hacia recinto Bustamante #20, veh??culo ser?? trasladado por personal equipo de mantenimiento el Sr. Johan Blancheteau ."

#DESBLOQUEO DE CUENTAS 
#text="Desbloqueo/reset No puede conectarse a la vpnSolicitud/ incidente: No puede conectarse a vpn Nombres: Luis Alejandro Galleguillos GalleguillosAnexo / Telefono: 942427492Correo: lgalleguillos@aguasdelvalle.clUsuario: lgalleguillosBloqueo / reset: "
#text = "Nombre: Javier LeguaEmail: jlegua@esval.clUsuario: jleguaBloqueo/Reset: ResetDescripcion: usuario indica que le llego correo para cambio de contrase??a "
#text = "Desbloqueo/reset Tiene el usuario bloqueadoSolicitud/ incidente:  desbloqueo de usuario Nombres: Roberto Enrique Stevenson AriasRut: 10889980-8Anexo / Telefono: 9929Correo: rstevenson@esval.clUsuario: Rstevenson Bloqueo / reset: Reset "

#Creacion de cuenta
#text = "POR FAVOR NECESITO CLAVE DE ACCESO AL SIGEC, SE BLOQUEO"
#text = "Estimados, Solicitamos por favor de su apoyo en la extensi??n de la cuenta del practicante Nicole Saez C."# - Pr??ctica <nsaez.practica@esval.cl> hasta el 22 de julio de 2022Lo anterior es porque Nicoles realizar?? su memoria. Gracias!"

#REPARACION
#text="Se requiere reponer o reparar tapas de wc de ba??os de "#hombres de 1"# norte"# Vi??a del Mar"
#text = "Estimado favor se requiere reparaci??n de llaves de jard??n en oficina PTAS La Chimba, y por filtraci??n. atte Constanza Oyanedel"
#text = "Estimado favor se requiere reparaci??n de llaves de jard??n en oficina PTAS La Chimba, y por filtraci??n. atte Constanza Oyanedel"

#REQUERIMIENTO
#text = " favor realizar gesti??n para env??o de equipamiento para : Usuario 	:Alexis Brian Gonz??lez Sep??lveda direcci??n 	: Paradero 3 1/2 San Pedro, Fundo El Molino, Quillotacontacto 	: +56987603675 se adjunta gu??a  de despacho : env??o "
#text = "Se solicita sanitizaci??n por afloramiento de aguas servidas en ba??o de iglesia. SISDA: 2372871 Direcci??n: Calle Huici # 585. Comuna: La Calera. Cliente: Guillermo Mart??nez Tel??fono de contacto: 9 88635656 Recinto: Ba??o ubicado en parte trasera de iglesia 9 m2 app."


# In[180]:


if st.button("Calcula"):
    t = normalize(x)
    #count_train_tfidf.fit(t)
    mt = count_train_tfidf.transform(t)
    #print(mt)

    topic_probability_scores = best_lda_model.transform(mt)
    #x = pd.DataFrame(topic_probability_scores).mean(axis=0)
    #x['Topic'] = Topic
    topic = df_topic_keywords.iloc[np.argmax(pd.DataFrame(topic_probability_scores).mean(axis=0)),1:9].values.tolist()
    infer_topic = df_topic_keywords.iloc[np.argmax(pd.DataFrame(topic_probability_scores).mean(axis=0)),-1]
    #topic_guess = df_topic_keywords.iloc[np.argmax(topic_probability_scores), Topic]
    st.text(infer_topic)


# In[ ]:





# In[ ]:




