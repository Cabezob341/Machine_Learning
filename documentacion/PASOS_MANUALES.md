# Guía completa de los pasos MANUALES en GitHub y Google Colab

La actividad exige **repositorio individual** y **notebooks GitHub → Colab**. Estos pasos requieren iniciar sesión con tus cuentas y no puedo hacerlos por ti.

## A. Crear el repositorio GitHub
1. Ingresa a https://github.com/ y accede a tu cuenta.
2. Haz clic en **+ → New repository**.
3. Escribe el nombre `challenge-ml-sueno-infantil` (puedes elegir otro).
4. Selecciona **Public**, salvo que el profesor haya indicado otra visibilidad; si necesitas privacidad, usa **Private** y otorga acceso al docente por el mecanismo correspondiente.
5. En **Description** escribe: `Pipeline ML — Caso C: sueño en los primeros meses de vida — UPCH`.
6. Puedes dejar desmarcada la opción de añadir README, ya que el paquete contiene uno completo.
7. Pulsa **Create repository**.
8. Descarga y descomprime el ZIP proporcionado. En la página del repositorio, selecciona **Add file → Upload files**.
9. Sube **el contenido** de la carpeta `Challenge_Sueno_ML` (README, notebooks, datos, imágenes y documentación). Si la interfaz no conserva carpetas al arrastrarlas, usa GitHub Desktop o el comando Git descrito al final.
10. En **Commit changes**, escribe `Agregar Challenge ML completo — caso sueño infantil` y confirma el commit.
11. Revisa que aparezcan `README.md`, `notebooks/`, `datos/`, `imagenes/` y `documentacion/`.

## B. Abrir y ejecutar Google Colab
1. Entra a https://colab.research.google.com/ con Google.
2. Usa **Archivo → Abrir cuaderno → GitHub**; pega la URL de tu repositorio y selecciona `01_Razonamiento_Busqueda.ipynb`.
3. Haz clic en **Entorno de ejecución → Ejecutar todo**. Espera a que aparezcan la búsqueda y el gráfico.
4. Repite con `02_Regresion_Sueno.ipynb`. Este notebook **genera internamente** el conjunto de datos con semilla fija; no necesitas montar Drive ni cargar el CSV.
5. Comprueba que aparezcan las métricas MAE, RMSE y R², junto a los gráficos de edad, predicción y residuos.
6. **Guarda los resultados ejecutados**: Archivo → Guardar una copia en GitHub (si la opción está disponible y autorizas acceso) o Archivo → Descargar → Descargar .ipynb y vuelve a cargarlo en el repositorio reemplazando la versión sin ejecutar.
7. Verifica en GitHub que los dos notebooks conserven las celdas de código, texto y gráficos de salida.

## C. Verificación final antes de enviar
- [ ] Nombre de repositorio correcto e individual.
- [ ] README describe el problema, justifica IA/ML y referencia la infografía.
- [ ] Espacio de estados incluye estados, operadores, restricciones, visual y objetivo.
- [ ] Notebook 1 ejecuta BFS y muestra solución.
- [ ] Notebook 2 ejecuta la regresión, métricas y gráficos.
- [ ] Datos etiquetados expresamente como sintéticos (NO datos reales del estudio).
- [ ] Enlace GitHub abre correctamente con los permisos adecuados.
- [ ] Si el docente pide Colab, copia también enlaces a los notebooks abiertos en Colab.
- [ ] Sube la foto original del examen solo si está permitido por el docente; **no es necesario** para ejecutar el código.

## Alternativa por línea de comandos (opcional)
```bash
git clone https://github.com/TU_USUARIO/challenge-ml-sueno-infantil.git
cd challenge-ml-sueno-infantil
# Copia aquí todo el contenido de la carpeta del proyecto
git add .
git commit -m "Agregar Challenge ML completo"
git push origin main
```

## Texto para presentar al docente
«El proyecto documenta el caso C del parcial sobre el sueño infantil y formaliza un espacio de estados con restricciones y búsqueda BFS. Se implementó un pipeline de aprendizaje supervisado mediante regresión lineal múltiple para estimar la duración del ciclo de sueño. Los datos de entrenamiento son sintéticos y se presentan por separado de los datos resumidos de la infografía. El modelo incorpora evaluación y gráficos interpretativos, y se ejecuta en Google Colab desde un repositorio individual de GitHub».
