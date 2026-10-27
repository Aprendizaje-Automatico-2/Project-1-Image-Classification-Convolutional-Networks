from pathlib import Path

from PIL import Image, ImageOps

from fishcls.config import EXTENSIONES_IMAGEN, TAMANO_PREPARADO


def listar_imagenes(carpeta: Path) -> list[tuple[Path, str]]:
    return [
        (ruta, clase.name)
        for clase in sorted(carpeta.iterdir())
        if clase.is_dir()
        for ruta in sorted(clase.iterdir())
        if ruta.suffix.lower() in EXTENSIONES_IMAGEN
    ]


def abrir_imagen(ruta: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")


def reducir(ruta: Path) -> tuple[Image.Image, int]:
    imagen = abrir_imagen(ruta)
    alto, ancho = TAMANO_PREPARADO
    return imagen.resize((ancho, alto), Image.Resampling.LANCZOS), imagen.width
