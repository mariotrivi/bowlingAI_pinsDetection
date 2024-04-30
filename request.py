import requests

# URL de la API a la que estás enviando la imagen
url = 'http://0.0.0.0:8000/upload/'

# Ruta al archivo de imagen que deseas enviar
file_path = 'input/people_car.jpg'

# Abre el archivo de imagen en modo binario
with open(file_path, 'rb') as image_file:
    # Define el archivo en un diccionario con la clave que espera la API (por ejemplo, 'file')
    files = {'file': image_file}

    # Realiza una solicitud POST a la API con el archivo de imagen
    response = requests.post(url, files=files)

    # Imprime la respuesta de la API
    print(response.text)

# Verifica si la solicitud fue exitosa
if response.status_code == 200:
    print("Imagen enviada con éxito.")
else:
    print("Error al enviar la imagen:", response.status_code)
