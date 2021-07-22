# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
from matplotlib import pyplot as plt 
import numpy as np
import argparse
import imutils
import cv2
import time
import math
import collections
import json
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle,ConnectionPatch

#obtenemos la matriz de transformacion de perspectiva
def obtenerMatriz(r,c):
	pts1 = np.float32([[297,150],[655,150],[845,465],[100,465]])
	pts2 = np.float32([[0,0],[r,0],[r,c],[0,c]])
	return cv2.getPerspectiveTransform(pts1, pts2)

def dibujarPuntos():
	original_points = np.float32([[297,150],[655,150],[845,465],[100,465]])
	src = cv2.imread('captura_campo_950.png')
	img_points = src.copy()
	for p in original_points:
		cv2.circle(img_points, center=tuple(p), radius=20, color=(0,255,0), thickness=-1)
	cv2.imshow("Puntos",img_points)
	cv2.waitKey()

#funcion para visualizar la imagen transformada 
def transformarPerspectiva(r,c):
	src = cv2.imread('captura_campo_950.png')
	#pts1 = np.float32([[385,190],[860,190],[133,600],[1120,600]]) original
	pts1 = np.float32([[297,150],[655,150],[845,465],[100,465]])
	pts2 = np.float32([[0,0],[r,0],[r,c],[0,c]])
	M = cv2.getPerspectiveTransform(pts1, pts2)
	dst = cv2.warpPerspective(src, M, (r,c))
	cv2.imshow('captura_campo_950.png', src)
	cv2.imshow('Transform', dst)
	cv2.waitKey()

#con la matriz de transformacion y el punto original, obtenemos el punto resultante
def calcularPunto(matriz,pt):
	#append de z=1
	pt = np.append(pt,1)
	#multiplicamos matriz por vector (x,y,z)
	pt = np.matmul(matriz,pt)
	#tenemos x'y'z', debemos eliminar z'
	pt = pt / pt[2]
	pt = np.delete(pt,2)
	return pt

#calculamos el numero de segundos que tiene el video
def calcularSegundosVideo():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
	help="path to the video file")
	args = vars(ap.parse_args())
	vs = cv2.VideoCapture(args["video"])

	numero_frames = float(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = float(vs.get(cv2.CAP_PROP_FPS))
	duracion = numero_frames / fps
	return int(math.ceil(duracion))

#calculamos la media de los valores que nos llegan en un conjunto/lista
def mediaValores(conjunto):
	s = 0
	for elemento in conjunto:
		s += elemento
	return s / float(len(conjunto))

#nos quedamos con los valores x de los pares (x,y) de una clave del diccionario
def valoresX(dict,clave):
	valores_x=[]
	for punto in dict[clave]:
		valores_x.append(punto[0])
	return valores_x

#nos quedamos con los valores y de los pares (x,y) de una clave del diccionario
def valoresY(dict,clave):
	valores_y=[]
	for punto in dict[clave]:
		valores_y.append(punto[1])
	return valores_y

#calculamos el punto medio de todas las coordenadas para todas las claves de un diccionario
def transformarValores(dict):

	for array in dict:
		if len(dict[array])!=0:
			media_x=mediaValores(valoresX(dict,array))
			media_y=mediaValores(valoresY(dict,array))
			dict[array]=[media_x, media_y]

#funcion para saber si tenemos algun punto erroneo, que se sale de nuestro rango/pista
def erroresRango(campo,dict,r,c):
	errores_x=0
	errores_y=0

	if campo=="arriba":

		for array in dict:
			if len(dict[array])!=0:
				e = dict[array]
				if (e[0]<0) or (e[0]>r):
					errores_x+=1
				if (e[1]<0) or (e[1]>(c/2)):
					errores_y+=1

		return errores_x,errores_y

	elif campo=="abajo":

		for array in dict:
			if len(dict[array])!=0:
				e = dict[array]
				if (e[0]<0) or (e[0]>r):
					errores_x+=1
				if (e[1]<(c/2)) or (e[1]>c):
					errores_y+=1

		return errores_x,errores_y
	else:
		raise ValueError("Debes indicar Arriba o abajo")


#funciones para eliminar coordenadas negativas e igualar a 0 si estan dentro de nuestro umbral

def getNeutralN(x):
  return [0 if x[0]<0 else x[0]
      ,0 if x[1]<0 else x[1]]

def umbralizarN(umbral,lista):
	newlist = [ getNeutralN(x) for x in[ [fst,snd] for [fst,snd] 
				in lista if (fst>umbral and snd>umbral) ]]
	return newlist

def eliminarN(umbral,dict):
	for array in dict:
		dict[array]=umbralizarN(umbral,dict[array])

#funciones para eliminar coordenadas demasiado positivas e igualar al punto maximo si estan dentro de nuestro umbral

def getNeutralP(x,r,c):
  return [r if x[0]>r else x[0]
      ,c if x[1]>c else x[1]]

def umbralizarP(umbral,r,c,lista):
	newlist = [ getNeutralP(x,r,c) for x in[ [fst,snd] for [fst,snd] 
				in lista if (fst<(r+umbral) and snd<(c+umbral)) ]]
	return newlist

def eliminarP(umbral,r,c,dict):
	for array in dict:
		dict[array]=umbralizarP(umbral,r,c,dict[array])


# funciones para obtener colores dominantes con kmeans

def centroid_histogram(clt):
  # grab the number of different clusters and create a histogram
  # based on the number of pixels assigned to each cluster
  numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
  (hist, _) = np.histogram(clt.labels_, bins = numLabels)

  # normalize the histogram, such that it sums to one
  hist = hist.astype("float")
  hist /= hist.sum()

  # return the histogram
  return hist

def plot_colors(hist, centroids):
  # initialize the bar chart representing the relative frequency
  # of each of the colors
  bar = np.zeros((50, 300, 3), dtype = "uint8")
  startX = 0

  # loop over the percentage of each cluster and the color of
  # each cluster
  for (percent, color) in zip(hist, centroids):
    # plot the relative percentage of each cluster
    endX = startX + (percent * 300)
    cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
      color.astype("uint8").tolist(), -1)
    startX = endX
  
  # return the bar chart
  return bar

def obtenerColores(imagen):

	imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
	imagen = imagen.reshape((imagen.shape[0] * imagen.shape[1], 3))
	clt = KMeans(n_clusters = 4)
	clt.fit(imagen)
	hist = centroid_histogram(clt)

	return (zip(hist, clt.cluster_centers_))


# function to get the output layer names in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    if label=="person":
        #color = COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0, 255, 0), 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#proceso general de deteccion de jugadores
def deteccion(pts,width,dict1,dict2,segundos):

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
	help="path to the video file")
	args = vars(ap.parse_args())

	vs = cv2.VideoCapture(args["video"])

	# allow the camera or video file to warm up
	time.sleep(2.0)

	#variables para realizar metricas
	fps = int(vs.get(cv2.CAP_PROP_FPS))
	jugador_izquierda_total=0 
	jugador_derecha_total=0
	frame_actual=0

	#variables para calcular puntos en la imagen transformada
	M = obtenerMatriz(300,600)

	num_detecciones_masde2=0 #variable para medir si existen mas de 2 detecciones simultaneas
	arrays_derecha = dict1
	arrays_izquierda = dict2

	# read pre-trained model and config file
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

	# keep looping
	while True:

		# grab the current frame
		frame = vs.read()
		frame_actual+=1

		# handle the frame from VideoCapture or VideoStream
		frame = frame[1] if args.get("video", False) else frame

		# if we are viewing a video and we did not grab a frame,
		# then we have reached the end of the video
		if frame is None:
			break

		# resize the frame, blur it, and convert it to the HSV
		# color space
		frame = imutils.resize(frame, width = width)

		#aplicaremos la mascara para prestar atencion a una unica parte del campo
		height,width,depth = frame.shape
		polylines_frame = np.zeros((height,width), np.uint8)

		cv2.fillPoly(polylines_frame, pts =[pts], color=(255,255,255))
		masked_data = cv2.bitwise_and(frame, frame, mask=polylines_frame)

		Width = masked_data.shape[1]
		Height = masked_data.shape[0]
		scale = 0.00392

		# create input blob
		blob = cv2.dnn.blobFromImage(masked_data, scale, (416,416), (0,0,0), True, crop=False)

		# set input blob for the network
		net.setInput(blob)

		# run inference through the network and gather predictions from output layers
		outs = net.forward(get_output_layers(net))

		# initialization
		class_ids = []
		confidences = []
		boxes = []
		conf_threshold = 0.5
		nms_threshold = 0.4

		# for each detetion from each output layer 
		# get the confidence, class id, bounding box params
		# and ignore weak detections (confidence < 0.5)
		for out in outs:
		    for detection in out:
		        scores = detection[5:]
		        class_id = np.argmax(scores)
		        confidence = scores[class_id]
		        if confidence > 0.5:
		            center_x = int(detection[0] * Width)
		            center_y = int(detection[1] * Height)
		            w = int(detection[2] * Width)
		            h = int(detection[3] * Height)
		            x = center_x - w / 2
		            y = center_y - h / 2
		            class_ids.append(class_id)
		            confidences.append(float(confidence))
		            boxes.append([x, y, w, h])

		# apply non-max suppression
		indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

		# contar el numero de detecciones en este frame y las almacenamos
		detecciones=0
		coordenadas_detecciones = {}
		eliminados=0
		
		for i in indices:
			i = i[0]
			box = boxes[i]
			x = box[0]
			y = box[1]
			w = box[2]
			h = box[3]

			# > para zona de arriba
			# < para zona de abajo
			if int(y+h)>height/2:
			 	continue
			else:
				coordenadas_detecciones[detecciones]=([int(x), int(y), int(x+w), int(y+h)])
				detecciones=detecciones+1

		# go through the detections remaining after nms and draw bounding box
		for i in indices:
			i = i[0]
			box = boxes[i]
			x = box[0]
			y = box[1]
			w = box[2]
			h = box[3]

			# > para zona de arriba
			# < para zona de abajo
			if int(y+h)>height/2:
			 	continue
			else:
				draw_prediction(masked_data, class_ids[i], confidences[i], int(x), int(y), int(x+w), int(y+h))
				
				if detecciones==1:
					punto_medio = int(x)+int(x+w)
					if ((punto_medio/2) < (width/2)):
						jugador_izquierda_total=jugador_izquierda_total+1

						pt = calcularPunto(M,np.array([punto_medio/2,int(y+h)]))
						n = (int(math.floor(frame_actual/fps)))

						arrays_izquierda[n].append(pt)

					else:
						jugador_derecha_total=jugador_derecha_total+1

						pt = calcularPunto(M,np.array([punto_medio/2,int(y+h)]))
						n = (int(math.floor(frame_actual/fps)))

						arrays_derecha[n].append(pt)

				elif detecciones==2:
					(x1, y1, x2, y2) = coordenadas_detecciones.get(0)
					(x3, y3, x4, y4) = coordenadas_detecciones.get(1)

					if (((x1+x2)/2) < (x3+x4/2)):

						jugador_izquierda_total=jugador_izquierda_total+1

						pt = calcularPunto(M,np.array([(x1+x2)/2,y2]))
						n = (int(math.floor(frame_actual/fps)))

						arrays_izquierda[n].append(pt)

						jugador_derecha_total=jugador_derecha_total+1

						pt = calcularPunto(M,np.array([(x3+x4)/2,y4]))
						n = (int(math.floor(frame_actual/fps)))

						arrays_derecha[n].append(pt)

					else:

						jugador_derecha_total=jugador_derecha_total+1

						pt = calcularPunto(M,np.array([(x1+x2)/2,y2]))
						n = (int(math.floor(frame_actual/fps)))

						arrays_derecha[n].append(pt)

						jugador_izquierda_total=jugador_izquierda_total+1

						pt = calcularPunto(M,np.array([(x3+x4)/2,y4]))
						n = (int(math.floor(frame_actual/fps)))

						arrays_izquierda[n].append(pt)

				else:
					num_detecciones_masde2=num_detecciones_masde2+1

		cv2.imshow("After NMS", masked_data)
		key = cv2.waitKey(1) & 0xFF

		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break

#para las coordenadas resultantes, posicionarlas en segun que zona para realizar graficas
def zonasCampo(dict,c,zona):

	red = 0
	fondo = 0
	zonap = 0

	if zona == "abajo":
		c1 = c/2
		c2 = (c+c1)/2
		c3 = c
		cp = ((c3+c2)/2)-(c/24)

		for array in dict:
			if len(dict[array])!=0:
				y = dict[array][1]
				if y>=c1 and y<=c2:
					red+=1
				elif y>c2 and y<=cp:
					zonap+=1
				elif y>cp and y<=c3:
					fondo+=1

	elif zona =="arriba":
		c1 = c/2
		c2 = c/4
		c3 = 0
		cp = ((c3+c2)/2)+(c/24)

		for array in dict:
			if len(dict[array])!=0:
				y = dict[array][1]
				if y<=c1 and y>=c2:
					red+=1
				elif y<c2 and y>=cp:
					zonap+=1
				elif y<cp and y>=c3:
					fondo+=1

	return [red, zonap, fondo]

#detectar cuantas veces y en que segundos el jugador ha estado en la "zona prohibida"
def zonaProhibida(dict,c,zona):
	#para c=600, campos de 300, 300-450 red, 450-600 fondo, zonaProhibida 450-500
	lista = []
	incrementos = c/12
	if zona=='abajo':
		limiteG = c - incrementos*2
		limiteP = c - incrementos*3
	elif zona=='arriba':
		limiteP = incrementos*2
		limiteG = incrementos*3

	for array in dict:
		if len(dict[array])!=0:
			y = dict[array][1]
			if y<limiteG and y>limiteP:
				lista.append(array)

	n = len(lista)
	return (n,lista)

# obtener posiciones incorrectas en el eje x, dos jugadores en el mismo lado de la pista
def posicionesIncorrectasX(dict_d,dict_i,r):
	mitad = r/2
	incorrectas_x = {}
	for array in dict_d:
		if len(dict_d[array])!=0 and len(dict_i[array])!=0:
			if (dict_d[array][0]<=mitad) and (dict_i[array][0]<=mitad):
				incorrectas_x[array] = True,'Izquierda'
			if (dict_d[array][0]>mitad) and (dict_i[array][0]>mitad):
				incorrectas_x[array] = True,'Derecha'
	return incorrectas_x

# obtener posiciones incorrectas en el eje y, dos jugadores en distintas zonas de la pista
def posicionesIncorrectasY(dict_d,dict_i,c,zona):
	incorrectas_y = {}

	if zona=='abajo':
		mitad = (c/2 + c)/2
	elif zona=='arriba':
		mitad = c/4
	
	for array in dict_d:
		if len(dict_d[array])!=0 and len(dict_i[array])!=0:
			if (dict_d[array][1]<=mitad) and (dict_i[array][1]>mitad):
				incorrectas_y[array] = True,'Descoordinacion'
			if (dict_d[array][1]>mitad) and (dict_i[array][1]<=mitad):
				incorrectas_y[array] = True,'Descoordinacion'
	return incorrectas_y

#claves vacias para los arrays, utilizamos esto para calcular porcentaje de deteccion .... etc.
def perdidas(dict):
	perdidas=0
	posiciones=[]
	for array in dict:
		if len(dict[array])==0:
			perdidas+=1
			posiciones.append(array)
	return (perdidas,posiciones)

#Funcion para agregar una etiqueta con el valor en cada barra
def autolabel(rects,subP):
    for rect in rects:
        height = rect.get_height()
        ax = subP
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

#construimos grafico de barras colectivo para comparar las zonas de desincronizacion.
def graficoBarras1(valores1,valores2,segundos):

	zonas = ['Descoordinacion Horizontal', 'Descoordinacion Vertical']
	a = (valores1 * 100)/float(segundos)
	b = (valores2 * 100)/float(segundos)
	desc = [round(a,2),round(b,2)]
	 
	fig, ax = plt.subplots()
	#Colocamos una etiqueta en el eje Y
	ax.set_ylabel('Porcentaxe de Tempo Descoordinados %')
	#Colocamos una etiqueta en el eje X
	ax.set_title('Zonas')
	#Creamos la grafica de barras utilizando 'paises' como eje X y 'ventas' como eje y.
	plt.bar(zonas, desc)
	plt.savefig('descoordinacion.png')
	#Finalmente mostramos la grafica con el metodo show()
	plt.show()

#construimos grafico de barras colectivo para comparar las zonas prohibidas
def graficoBarras3(valores1,valores2,segundos):

	jugadores = ['Xogador Esquerda', 'Xogador Dereita']
	a = (valores1[0] * 100)/float(segundos)
	b = (valores2[0] * 100)/float(segundos)
	zonas_prohibidas = [round(a,2),round(b,2)]
	 
	fig, ax = plt.subplots()
	#Colocamos una etiqueta en el eje Y
	ax.set_ylabel('Porcentaxe de Tempo Zona Prohibida %')
	#Colocamos una etiqueta en el eje X
	ax.set_title('Xogadores')
	#Creamos la grafica de barras utilizando 'paises' como eje X y 'ventas' como eje y.
	plt.bar(jugadores, zonas_prohibidas)
	plt.savefig('zonaprohibida.png')
	#Finalmente mostramos la grafica con el metodo show()
	plt.show()

#construimos grafico de barras para las zonas de la pista en la que esta un jugador
def graficoBarras2(valores1,valores2,segundos):

	asistencia = ['Pegado a rede', 'Zona Prohibida', 'Fondo da pista']

	a = (valores1[0]*100)/float(segundos)
	b = (valores1[1]*100)/float(segundos)
	c = (valores1[2]*100)/float(segundos)

	d = (valores2[0]*100)/float(segundos)
	e = (valores2[1]*100)/float(segundos)
	f = (valores2[2]*100)/float(segundos)

	jugador_izq = [round(a,2),round(b,2),round(c,2)]
	jugador_der = [round(d,2),round(e,2),round(f,2)]

	#Obtenemos la posicion de cada etiqueta en el eje de X
	x = np.arange(len(asistencia))
	#tamano de cada barra
	width = 0.35
	 
	fig, ax = plt.subplots()
	 
	#Generamos las barras para el conjunto de hombres
	rects1 = ax.bar(x - width/2, jugador_izq, width, label='Xogador Esquerda')
	#Generamos las barras para el conjunto de mujeres
	rects2 = ax.bar(x + width/2, jugador_der, width, label='Xogador Dereita')
	 
	#Anadimos las etiquetas de identificacion de valores en el grafico
	ax.set_ylabel('Porcentaxe de tempo %')
	ax.set_title('Zonas da pista')
	ax.set_xticks(x)
	ax.set_xticklabels(asistencia)
	#Anadimos un legen() esto permite mmostrar con colores a que pertence cada valor.
	ax.legend()
	 
	#Anadimos las etiquetas para cada barra
	autolabel(rects1,ax)
	autolabel(rects2,ax)
	fig.tight_layout()
	plt.savefig('zonas_comparacion.png')
	#Mostramos la grafica con el metodo show()
	plt.show()

#sumamos todos los enteros de una lista
def sumaLista(listaNumeros):
	if len(listaNumeros) == 0:
		return 0
	elif len(listaNumeros) == 1:
		return listaNumeros[0]
	else:
		return listaNumeros[0] + sumaLista(listaNumeros[1:])

#construimos grafico pastel
def graficoPastel(valores,jugador):

	zonas_campo = 'Pegado a rede', 'Zona prohibida', 'Fondo da pista'
	#Declaramos el tamano de cada 'rebanada' y en sumatoria todos deben dar al 100%
	
	size = sumaLista(valores)
	for e in valores:
		e = (e/size)*100 

	#En este punto senalamos que posicion debe 'resaltarse' y el valor, si se coloca 0, se omite
	maxpos = valores.index(max(valores))
	if maxpos==0:
		explode = (0.1, 0, 0)
	elif maxpos==1:
		explode = (0, 0.1, 0)
	else:
		explode = (0, 0, 0.1)

	fig1, ax1 = plt.subplots()
	#Creamos el grafico, anadiendo los valores
	ax1.pie(valores, explode=explode, labels=zonas_campo, autopct='%1.1f%%',
	        shadow=True, startangle=90)
	#senalamos la forma, en este caso 'equal' es para dar forma circular
	ax1.axis('equal')
	plt.title("Zonas do campo XOGADOR "+ jugador)
	plt.legend()
	plt.savefig('grafica_pastel_'+jugador+'.png')
	plt.show()

#le damos la vuelta al eje Y para utilizar las coordenadas en el grafico mapa de calor
def revertirEjeY(dict,c):
	for array in dict:
		if len(dict[array])!=0:
			y = dict[array][1]
			dict[array][1] = c-y

#separamos las coordenadas X e Y de los puntos para el mapa de calor
def prepareHM(dict):
	x = []
	y = []
	for array in dict:
		if len(dict[array])!=0:
			x.append(dict[array][0])
			y.append(dict[array][1])
	return x,y

#dibujamos una pista de padel en nuestros graficos
def draw_pitch(ax,b):
    Pitch = Rectangle([0,0], width = 300, height = 600, fill = b, color = 'green')
    red = ConnectionPatch([0,300], [300,300], "data", "data", color = 'black')
    rightline = ConnectionPatch([0,90], [300,90], "data", "data", color = 'black')
    leftline = ConnectionPatch([0,510], [300,510], "data", "data", color = 'black')
    centerline = ConnectionPatch([150,90], [150,510], "data", "data", color = 'black')
    
    element = [Pitch, rightline, leftline, centerline, red]
    for i in element:
        ax.add_patch(i)

def heatMatrix_aux(n,r,c,dic):

	matriz = np.zeros((n, n))

	for array in dic:
		if len(dic[array])!=0:

			x = dic[array][0]
			y = dic[array][1]-(c/2)
			fila = int(y/(r/n)) 
			columna = int(x/(r/n))

			if fila==n:
				fila = fila-1
			if columna==n:
				columna = columna-1
			matriz[fila,columna]+=1

	num_elementos = np.sum(matriz)

	for i in range(len(matriz)):
		for j in range(len(matriz[i])):
			matriz[i][j] = matriz[i][j]/num_elementos

	return matriz

# mapa de calor a traves de una matriz de porcentajes
def heatMatrix(n,r,c,dic):

	fig, ax = plt.subplots()
	draw_pitch(ax,False)

	ax.set_aspect('equal')
  	ax.set_title("Xogador Esquerda")
  	ax.set_xlim(0, r)
  	ax.set_ylim(0, c)

  	matriz = heatMatrix_aux(n,r,c,dic)
  	matriz = cv2.resize(matriz, (300,300))
  	#plt.matshow(matriz)
  	heatmapshow = None
	heatmapshow = cv2.normalize(matriz, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
	#cv2.imshow("Heatmap", heatmapshow)
  	
  	plt.imshow(cv2.cvtColor(heatmapshow, cv2.COLOR_BGR2RGB))
  	#plt.savefig('heatmap1.png')
  	plt.savefig('heatmap_bueno.png')
	plt.show()
	cv2.waitKey(0)

# escribir puntos a un txt
def escribir(archivo,diccionario):
	str(file)
	with open(archivo, "w") as archivo:
	    for (array, posiciones) in diccionario.items():
	        archivo.write("%s %s\n" %(array,posiciones))

# leer puntos de un txt
def leer(archivo):
	d = {}
	with open(archivo) as f:
		for line in f:
			contenido = line.split(None,1)
			clave = int(contenido[0])
			coordenadas = json.loads(contenido[1])
			if len(coordenadas)!=0:
				x = coordenadas[0]
				y = coordenadas[1]
				d[clave]= [x,y]
			else:
				d[clave]= []
	return d

def main():

	# read class names from text file
	global classes 
	classes = None
	with open("yolov3.txt", 'r') as f:
		classes = [line.strip() for line in f.readlines()]

   #crear array para cada segundo del video (distinguiendo entre jugador izquierda y derecha)
	segundos = calcularSegundosVideo()
	arrays_izquierda = {}
	arrays_derecha = {}

	for i in range(segundos):
	  arrays_izquierda[i] = []
	  arrays_derecha[i] = []

	#mascara y ancho de la parte inferior
	pts_abajo = np.array([[220,180],[730,180],[735,250],[875,465],[80,465],[220,240]], np.int32)
	width_abajo = 950
	#mascara y ancho de la parte superior
	pts_arriba = np.array([[250,100],[680,100],[730,340],[180,340]], np.int32)
	width_arriba = 950

	width_arriba2 = 1600
	pts_arriba2 = np.array([[480,150],[1100,150],[1250,400],[350,400]], np.int32)

	#deteccion para zona superior de la pista
	#deteccion(pts_arriba,width_arriba,arrays_derecha,arrays_izquierda,segundos)

	#deteccion para zona inferior de la pista
	#deteccion(pts_abajo,width_abajo,arrays_derecha,arrays_izquierda,segundos)

	#eliminarN(-10.01,arrays_derecha)
	#eliminarN(-10.01,arrays_izquierda)
	#eliminarP(10,300,600,arrays_derecha)
	#eliminarP(10,300,600,arrays_izquierda)

	transformarValores(arrays_derecha)
	transformarValores(arrays_izquierda)

	#escribir los resultados del proceso de deteccion si queremos guardarlos
	#escribir("derecha",arrays_derecha)
	#escribir("izquierda",arrays_izquierda)

	#leer resultados de un proceso de deteccion 
	arrays_derecha = leer("derecha")
	arrays_izquierda = leer("izquierda")

	valores1 = zonasCampo(arrays_derecha,600,"abajo")
	valores2 = zonasCampo(arrays_izquierda,600,"abajo")
	graficoBarras2(valores1,valores2,segundos)

	valores3 = len(posicionesIncorrectasX(arrays_derecha,arrays_izquierda,300))
	valores4 = len(posicionesIncorrectasY(arrays_derecha,arrays_izquierda,600,'abajo'))
	graficoBarras1(valores3,valores4,segundos)

	valores5 = zonaProhibida(arrays_derecha,600,'abajo')
	valores6 = zonaProhibida(arrays_izquierda,600,'abajo')
	graficoBarras3(valores5,valores6,segundos)

	graficoPastel(valores1, "DEREITA")
	graficoPastel(valores2, "ESQUERDA")

	#heatmaps
	heatMatrix(6,300,600,arrays_izquierda)
	heatMatrix(6,300,600,arrays_derecha)

	# close all windows
	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
