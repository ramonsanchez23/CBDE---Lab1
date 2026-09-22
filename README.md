# Proyecto CBDE - Lab1

## 1. Requisitos previos
  Python 3.10+ instalado.
  PostgreSQL instalado y en ejecución localmente.

## 2. Configuración del entorno de Python
Abre tu terminal en la carpeta del proyecto y sigue estos pasos:
  
  Paso 1: Crear y activar un entorno virtual
    En Windows (PowerShell):
      python -m venv venv
      .\venv\Scripts\Activate.ps1
      (Si te da error de ejecución de scripts en PowerShell, ejecuta primero: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process)
      
    En macOS/Linux:
      python3 -m venv venv
      source venv/bin/activate
    
  Paso 2: Instalar las librerías necesarias
    Con el entorno virtual activado, instala todas las dependencias del proyecto ejecutando esto:
      pip install -r requirements.txt
      
  Paso 3. Configuración de credenciales (PostgreSQL)
    Crea un archivo llamado database.ini en la raíz del proyecto (al mismo nivel que README.md) y añade tus datos de conexión a PostgreSQL con el siguiente formato pero con tus credenciales:
      [postgresql]
      host=localhost
      database=nombre_de_tu_bd
      user=tu_usuario_postgres
      password=tu_contraseña
      port=5432
