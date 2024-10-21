import csv

data = [
['0.05', '18.208557'],
['0.10', '6.1770462'],
['0.15', '6.1995073'],
['0.20', '7.2764875'],
['0.25', '10.133961'],
['0.30', '13.974145'],
['0.35', '20.369444'],
['0.40', '34.918618'],
['0.45', '38.672279'],
['0.50', '50.602469'],
['0.55', '68.589883'],
['0.60', '84.039895'],
['0.65', '119.49594'],
['0.70', '150'],
['0.75', '210'],
['0.80', '549'],
['0.85', '720'],
['0.90', '900'],
['0.95', '1200'],
]

with open('output.csv', mode = 'w', newline='') as file:
    writer = csv.writer(file)


    writer.writerows(data)

print("Archivo CSV creado exitosamente.")
import os
print(os.getcwd())  # Esto imprime el directorio actual de trabajo



