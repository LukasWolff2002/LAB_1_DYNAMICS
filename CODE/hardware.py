#En esa seccion se definen las propiedades de los componentes de la placa

#Material
E = 200000 #MPa
densidad = 7850 #kg/m3

#Sensores
Frecuencia_muestreo = 300 #Hz


#Barra 
lbarra = 42.5/100 #m, longitud de la barra
mbarra = 324.6/1000 #Kg, masa total de la barra

#Union barra masa sup
munion = 100/1000 #Kg ARREGLAR0

#Masa puntual Superior
msuperior = 100/1000 #Kg, ARREGLAR

#Sensor
msensonr = 13 #Kg, masa del sensor

#Calculo de inercias. REVISAR
Ibarra = (1/3)*mbarra*(lbarra**2) #Inercia de la barra
Iunion = (1/3)*munion*(lbarra**2) #Inercia de la union
Isuperior = (1/3)*(msuperior + msensonr)*(lbarra**2) #Inercia de la masa superior incluyendo el sensor

It = Ibarra + Iunion + Isuperior #Inercia total

