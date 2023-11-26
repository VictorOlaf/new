import json
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer,TFAutoModelForQuestionAnswering


#Su modelo de Clasificador
ruta_del_modelo = 'drive/MyDrive/ChatBot1'
tokenizerB = AutoTokenizer.from_pretrained(ruta_del_modelo)
model_intenciones = TFAutoModelForSequenceClassification.from_pretrained(ruta_del_modelo)

#Su modelo de Question Answering
tokenizerA = AutoTokenizer.from_pretrained("IIC/roberta-base-spanish-sqac")
model_respuestas = TFAutoModelForQuestionAnswering.from_pretrained("IIC/roberta-base-spanish-sqac")


with open(ruta_del_modelo+'/DiccionarioClases.json', 'r') as f:
    diccionario_cargado = json.load(f)

def predict_intention(text):
    inputs = tokenizerB(text, return_tensors="tf")
    predictions = model_intenciones(inputs['input_ids'])
    print(predictions.logits)
    return np.argmax(predictions.logits)

print(diccionario_cargado)

while True:
  pregunta = input("Usuario: ")
  if pregunta=="Detener":
    break
  intencion = str(predict_intention(pregunta))
  #print(f"Intención detectada: {diccionario_cargado[intencion]}")

  if diccionario_cargado[intencion]=="Saludo":
    Saludos=["Hola", "Hi", "Que hay"]
    eleccion=np.random.randint(0,len(Saludos))
    print(f"Asistente: {Saludos[eleccion]}")
  elif diccionario_cargado[intencion]=="Consulta de Precio":
    Contexto="un pastel de chocolate cuesta 20 pesos, un pastel de vainilla cuesta 30 pesos y un pastel de  "
    question, text = pregunta, Contexto
    inputs = tokenizerA(question, text, return_tensors="tf")
    outputs = model_respuestas(inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start_index = tf.argmax(start_scores, axis=1).numpy()[0]
    end_index = tf.argmax(end_scores, axis=1).numpy()[0] + 1

    respuesta = tokenizerA.convert_tokens_to_string(tokenizerA.convert_ids_to_tokens(inputs["input_ids"][0, start_index:end_index]))
    print(f"Asistente: {respuesta}")
  elif diccionario_cargado[intencion]=="Consulta de Envío":
    Contexto="El envio dentro de la ciudad de puebla cuesta 50 pesos, si es a otro estado de la republica el envio cuesta 100 pesos"
    question, text = pregunta, Contexto
    inputs = tokenizerA(question, text, return_tensors="tf")
    outputs = model_respuestas(inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start_index = tf.argmax(start_scores, axis=1).numpy()[0]
    end_index = tf.argmax(end_scores, axis=1).numpy()[0] + 1

    respuesta = tokenizerA.convert_tokens_to_string(tokenizerA.convert_ids_to_tokens(inputs["input_ids"][0, start_index:end_index]))
    print(f"Asistente: {respuesta}")
  elif diccionario_cargado[intencion]=="Despedida":
    marcador=0
    pregunta=pregunta.split(" ")
    for palabra in pregunta:
      if palabra=="Pablo":
        marcador=1
      elif palabra=="Pedro":
        marcador=2
    if marcador==1:
      print(f"Asistente: Adios Pablo")
    elif marcador==2:
      print(f"Asistente: Adios Pedro")
    else:
      print(f"Asistente: Adios Quien seas")
