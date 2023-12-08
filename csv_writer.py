import csv

# Datos de la fila que quieres escribir
datos_fila = ['fars', 'distance_fars', 'finger_tops', 'distance_fingers_top', 'img', 'gesture']

# Ruta del archivo CSV
ruta_csv = 'imgs.csv'

# Modo 'a' para abrir el archivo en modo de añadir (si ya existe) o crear uno nuevo
with open(ruta_csv, mode='a', newline='') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)

    # Escribir la fila en el archivo
    escritor_csv.writerow(datos_fila)
