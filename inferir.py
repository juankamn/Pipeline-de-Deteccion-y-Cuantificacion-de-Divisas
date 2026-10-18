"""Cuenta monedas y billetes en una foto, un video o la webcam.

No importa el notebook. Usa los pesos de pesos/monedas_billetes.pt.
"""
import argparse
import os
import statistics
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
CLASS_VALUE = {
    "moneda_50": 50,
    "moneda_100": 100,
    "moneda_200": 200,
    "moneda_500": 500,
    "moneda_1000": 1000,
    "billete_2000": 2000,
    "billete_5000": 5000,
    "billete_10000": 10000,
    "billete_20000": 20000,
    "billete_50000": 50000,
    "billete_100000": 100000,
}
CLASS_NAMES = list(CLASS_VALUE)
EXTENSIONES_FOTO = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def total_pesos(conteos):
    return sum(CLASS_VALUE[nombre] * int(cantidad) for nombre, cantidad in conteos.items() if cantidad > 0)


def texto_escena(conteos):
    partes = []
    for nombre in CLASS_NAMES:
        cantidad = int(conteos.get(nombre, 0))
        if cantidad <= 0:
            continue
        valor = CLASS_VALUE[nombre]
        if nombre.startswith("billete_"):
            unidad = "billete" if cantidad == 1 else "billetes"
        else:
            unidad = "moneda" if cantidad == 1 else "monedas"
        partes.append(f"{cantidad} {unidad} de {valor}")
    total = total_pesos(conteos)
    if not partes:
        return f"No se detectó dinero. Total en escena: ${total}"
    if len(partes) == 1:
        lista = partes[0]
    else:
        lista = ", ".join(partes[:-1]) + f" y {partes[-1]}"
    return f"Se detectaron {lista}. Total en escena: ${total}"


def contar_nombres(nombres):
    conteos = {nombre: 0 for nombre in CLASS_NAMES}
    for nombre in nombres:
        if nombre in conteos:
            conteos[nombre] += 1
    return conteos


def conteo_predicho(resultado):
    nombres = []
    if resultado.boxes is not None:
        for clase in resultado.boxes.cls.tolist():
            nombres.append(resultado.names[int(clase)])
    return contar_nombres(nombres)


def conteo_mediana(ventana):
    recientes = ventana[-15:]
    estable = {}
    for nombre in CLASS_NAMES:
        estable[nombre] = int(round(statistics.median([int(conteo[nombre]) for conteo in recientes])))
    return estable


def vacio():
    return {nombre: 0 for nombre in CLASS_NAMES}


def imgsz_del_frame(ancho, alto):
    return 832 if max(ancho, alto) <= 928 else 1024


def tamano_salida(ancho, alto, tope=960):
    escala = min(1.0, tope / max(ancho, alto))
    return max(2, int(ancho * escala) // 2 * 2), max(2, int(alto * escala) // 2 * 2)


def elegir_device(pedido):
    if pedido:
        return pedido
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


MARGEN_FRANJA = 18
ALTO_LINEA = 36
LINEAS_FRANJA = 3
ESCALA_TEXTO = 0.75
GROSOR_TEXTO = 2


def alto_franja():
    return MARGEN_FRANJA * 2 + ALTO_LINEA * LINEAS_FRANJA


def lineas_dinero(conteos):
    def frase(prefijo, etiqueta, vacio):
        partes = []
        for nombre in CLASS_NAMES:
            if not nombre.startswith(prefijo):
                continue
            cantidad = int(conteos.get(nombre, 0))
            if cantidad <= 0:
                continue
            partes.append(f"{cantidad} de {CLASS_VALUE[nombre]}")
        if not partes:
            return f"{etiqueta}: {vacio}."
        return f"{etiqueta}: " + ", ".join(partes) + "."

    return [
        frase("moneda_", "Monedas", "ninguna"),
        frase("billete_", "Billetes", "ninguno"),
        f"Total en escena: ${total_pesos(conteos)}",
    ]


def partir_texto(texto, ancho_px):
    palabras = texto.split()
    lineas = []
    actual = ""
    for palabra in palabras:
        candidato = palabra if not actual else f"{actual} {palabra}"
        ancho_texto = cv2.getTextSize(candidato, cv2.FONT_HERSHEY_SIMPLEX, ESCALA_TEXTO, GROSOR_TEXTO)[0][0]
        if ancho_texto <= ancho_px or not actual:
            actual = candidato
        else:
            lineas.append(actual)
            actual = palabra
    if actual:
        lineas.append(actual)
    return lineas


def agregar_franja(imagen, conteos):
    alto, ancho = imagen.shape[:2]
    lineas = []
    for frase in lineas_dinero(conteos):
        lineas.extend(partir_texto(frase, max(40, ancho - 24)))
    lineas = lineas[:LINEAS_FRANJA]
    lienzo = cv2.copyMakeBorder(imagen, 0, alto_franja(), 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    y = alto + MARGEN_FRANJA + 24
    for linea in lineas:
        cv2.putText(
            lienzo, linea, (12, y), cv2.FONT_HERSHEY_SIMPLEX, ESCALA_TEXTO, (0, 0, 0), GROSOR_TEXTO, cv2.LINE_AA,
        )
        y += ALTO_LINEA
    return lienzo


def mostrar(ventana, imagen):
    if os.environ.get("INFERIR_SIN_ESPERA") == "1":
        return False
    try:
        cv2.imshow(ventana, imagen)
        return True
    except cv2.error:
        print("No hay ventana grafica. La frase queda en la consola.")
        return False


def ruta_guardado(fuente, guardar, es_video):
    if not guardar:
        return None
    if guardar == "auto":
        base = ROOT / "modelo_extras" / "inferir"
        base.mkdir(parents=True, exist_ok=True)
        nombre = "webcam" if not isinstance(fuente, Path) else fuente.stem
        sufijo = ".mp4" if es_video else ".jpg"
        return base / f"{nombre}_conteo{sufijo}"
    destino = Path(guardar)
    if destino.suffix == "":
        destino = destino.with_suffix(".mp4" if es_video else ".jpg")
    destino.parent.mkdir(parents=True, exist_ok=True)
    return destino


def abrir_salida(destino, fps, ancho, alto):
    for codigo in ("avc1", "mp4v"):
        salida = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*codigo), fps, (ancho, alto))
        if salida.isOpened():
            return salida
        salida.release()
    raise RuntimeError(f"No pude crear el video {destino}")


def inferir_foto(modelo, ruta, imgsz, conf, device, destino):
    resultado = modelo.predict(str(ruta), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    conteo = conteo_predicho(resultado)
    frase = texto_escena(conteo)
    print(frase)
    imagen = agregar_franja(resultado.plot(), conteo)
    if destino is not None:
        cv2.imwrite(str(destino), imagen)
        print("Guardada:", destino)
    if mostrar("conteo", imagen) and sys.stdin.isatty():
        print("Cierra la ventana o pulsa una tecla para salir.")
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def inferir_video(modelo, captura, imgsz, conf, device, destino, titulo):
    fps = captura.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 20
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if imgsz <= 0:
        imgsz = imgsz_del_frame(ancho, alto)
    salida_w, salida_h = tamano_salida(ancho, alto)
    salida = None if destino is None else abrir_salida(destino, fps, salida_w, salida_h + alto_franja())
    ventana = []
    hay_ventana = True
    print("Pulsa q para salir. imgsz", imgsz)
    while True:
        ok, frame = captura.read()
        if not ok:
            break
        resultado = modelo.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        ventana.append(conteo_predicho(resultado))
        if len(ventana) > 15:
            ventana.pop(0)
        estable = conteo_mediana(ventana)
        imagen = resultado.plot()
        if imagen.shape[1] != salida_w or imagen.shape[0] != salida_h:
            imagen = cv2.resize(imagen, (salida_w, salida_h))
        imagen = agregar_franja(imagen, estable)
        if salida is not None:
            salida.write(imagen)
        if hay_ventana:
            hay_ventana = mostrar(titulo, imagen)
        if hay_ventana and (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    captura.release()
    if salida is not None:
        salida.release()
        print("Video guardado:", destino)
    print(texto_escena(conteo_mediana(ventana) if ventana else vacio()))
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Cuenta monedas y billetes en una foto, un video o la webcam.")
    parser.add_argument("--fuente", required=True, help="Ruta de una imagen, de un video, o el indice de la webcam (0)")
    parser.add_argument("--pesos", default=str(ROOT / "pesos" / "monedas_billetes.pt"))
    parser.add_argument("--imgsz", type=int, default=0, help="0 elige 640 en una foto y 832 o 1024 en video")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--device", default="", help="Vacio: GPU si esta disponible, si no CPU")
    parser.add_argument("--guardar", nargs="?", const="auto", default="", help="Ruta de salida. Solo el flag guarda en modelo_extras/inferir")
    args = parser.parse_args()

    pesos = Path(args.pesos)
    if not pesos.exists():
        raise SystemExit(f"No estan los pesos: {pesos}")
    device = elegir_device(args.device)

    from ultralytics import YOLO

    modelo = YOLO(str(pesos))
    fuente = args.fuente
    if fuente.isdigit():
        captura = cv2.VideoCapture(int(fuente))
        if not captura.isOpened():
            raise SystemExit(f"No pude abrir la webcam {fuente}")
        destino = ruta_guardado(fuente, args.guardar, True)
        inferir_video(modelo, captura, args.imgsz, args.conf, device, destino, "webcam")
        return

    ruta = Path(fuente)
    if not ruta.exists():
        raise SystemExit(f"No existe la fuente: {ruta}")
    if ruta.suffix.lower() in EXTENSIONES_FOTO:
        imgsz = args.imgsz if args.imgsz > 0 else 640
        inferir_foto(modelo, ruta, imgsz, args.conf, device, ruta_guardado(ruta, args.guardar, False))
        return

    captura = cv2.VideoCapture(str(ruta))
    if not captura.isOpened():
        raise SystemExit(f"No pude abrir el video: {ruta}")
    inferir_video(modelo, captura, args.imgsz, args.conf, device, ruta_guardado(ruta, args.guardar, True), ruta.name)


if __name__ == "__main__":
    main()
