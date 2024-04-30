from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from mancar_main import main as process_folder
import shutil

app = FastAPI()

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Aseguramos que borramos la carpeta de iteraciones anteriores
        upload_folder = "./uploaded_files"
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
        os.makedirs(upload_folder, exist_ok=True)

        # Guardar el archivo en el sistema de archivos local
        file_location = f"./uploaded_files/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        
        # Procesar la carpeta y obtener el diccionario JSON
        json_dict = process_folder("./uploaded_files/")
        
        # Devolver la respuesta JSON con el contenido procesado
        return JSONResponse(status_code=200, content=json_dict)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {e}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
