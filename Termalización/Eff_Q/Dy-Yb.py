import csv

data = [
['0.05', '212.94194'],
['0.10', '171.46854'],
['0.15', '178.91001'],
['0.20', '248.98845'],
['0.25', '294.98288'],
['0.30', '407.50924'],
['0.35', '445.50811'],
['0.40', '645.78804'],
['0.45', '857.67989'],
['0.50', '1146.4117'],
['0.55', '1902.9816'],
['0.60', '3199.7848'],
['0.65', '3130.9081']
]

with open('output.csv', mode = 'w', newline='') as file:
    writer = csv.writer(file)


    writer.writerows(data)

print("Archivo CSV creado exitosamente.")
import os
print(os.getcwd())  # Esto imprime el directorio actual de trabajo
