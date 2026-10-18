# Conteo de monedas y billetes

Detector YOLO que lee la denominación de cada pieza y suma el dinero de la escena. El entregable visual es `monedas_conteo_extras.ipynb`. La inferencia en este computador corre con `inferir.py` y los pesos de `pesos/`.

## Notebook

`monedas_conteo_extras.ipynb` ya está ejecutado. Ahí está el ciclo del taller:

- Secciones 2 y 9: muestras etiquetadas y el efecto de la iluminación.
- Sección 4: curvas de entrenamiento.
- Secciones 5, 6 y 11: cajas sobre las fotos y la validación del conteo. La frase es del tipo "Se detectaron 3 monedas de 100 y 1 de 500. Total en escena: $800".
- Sección 12: los tres videos, con la suma estable.
- Sección 13: conclusiones sobre falsos positivos, falsos negativos y oclusión.

Los videos ya contados se pueden abrir sin clonar el repositorio:

- [Monedas](https://github.com/juankamn/Pipeline-de-Deteccion-y-Cuantificacion-de-Divisas/blob/main/modelo_extras/videos/video_01_monedas_quitar_y_agregar.mp4)
- [Billetes](https://github.com/juankamn/Pipeline-de-Deteccion-y-Cuantificacion-de-Divisas/blob/main/modelo_extras/videos/video_02_billetes_quitar_y_agregar.mp4)
- [Monedas y billetes](https://github.com/juankamn/Pipeline-de-Deteccion-y-Cuantificacion-de-Divisas/blob/main/modelo_extras/videos/video_03_monedas_y_billetes_quitar_y_agregar.mp4)

## Datasets

Las fotos no están en este repositorio.

- Monedas: [monedas-v3](https://app.roboflow.com/pandebono-predictors-workspace/monedas-v3/3)
- Billetes: [billetes colombianos](https://app.roboflow.com/pandebono-predictors-workspace/billetes-colombianos-lgn8w/1)

Las fotos de entrenamiento no están en este repositorio: se descargan de esos dos enlaces. Las ocho fotos y los tres videos con los que se probó el modelo sí están en `fotos_propias/`. Con eso se pueden correr la sección 11, la sección 12 y `inferir.py`.

## Inferencia local

En esta máquina el entorno es Python 3.14 y PyTorch 2.12 con CUDA 12.6. `pip install torch` sin el índice de CUDA puede dejar la versión de CPU. Instala primero el wheel CUDA de PyTorch y después:

```bash
pip install -r requirements.txt
```

Los pesos por defecto son `pesos/monedas_billetes.pt`. La misma red exportada está en `pesos/monedas_billetes.torchscript`.

```bash
python inferir.py --fuente mi_foto.jpg
python inferir.py --fuente mi_video.mp4 --device cpu
python inferir.py --fuente 0
python inferir.py --fuente mi_foto.jpg --guardar
```

`--fuente` acepta una imagen, un video o el índice de la webcam. `--imgsz 0` usa 640 en una foto y 832 o 1024 en video. `--conf` parte en 0.40. `--device` vacío usa la GPU si hay, y si no la CPU. `--guardar` escribe el resultado en `modelo_extras/inferir/`. Esa carpeta no va al repositorio.

En video y en la webcam el conteo es la mediana de cada denominación en las últimas 15 frames, para que el total no salte de un frame a otro. Debajo del cuadro, una franja blanca lista las monedas, los billetes y el total. `q` cierra la ventana.
