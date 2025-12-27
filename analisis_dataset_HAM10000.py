import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
HAM_DIR = "HAM10000_images"
HAM_META = "HAM10000_metadata.csv"

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def mostrar_distribucion(conteo, titulo):
    total = sum(conteo.values())
    print(f"\n📊 {titulo}")
    for clase, cantidad in conteo.items():
        print(f" - {clase:25s}: {cantidad:5d} ({(cantidad/total)*100:.2f}%)")
    print(f"Total de imágenes: {total}\n")

    plt.figure(figsize=(8,4))
    plt.bar(conteo.keys(), conteo.values(), color="skyblue")
    plt.title(titulo)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cantidad de imágenes")
    plt.tight_layout()
    plt.show()

def analizar_tamanio_promedio(base_dir):
    tamanios = []
    for f in os.listdir(base_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                with Image.open(os.path.join(base_dir, f)) as img:
                    tamanios.append(img.size)
            except:
                continue
    if tamanios:
        anchos = [t[0] for t in tamanios]
        altos = [t[1] for t in tamanios]
        return round(sum(anchos)/len(anchos), 1), round(sum(altos)/len(altos), 1)
    return 0, 0

# =============================================================================
# ANALISIS HAM10000
# =============================================================================
def analizar_ham10000():
    print("🧠 Analizando dataset HAM10000...")
    df = pd.read_csv(HAM_META)
    total = len(df)
    print(f"Total de registros: {total}")

    conteo_clases = Counter(df["dx"])
    mostrar_distribucion(conteo_clases, "HAM10000 - Distribución por clase")

    # Clasificación Benigna/Maligna
    benignas = ["nv", "bkl", "df", "vasc"]
    malignas = ["mel", "bcc", "akiec"]

    print("\n🩺 Clasificación de lesiones:")
    print(f" - Benignas: {', '.join(benignas)}")
    print(f" - Malignas: {', '.join(malignas)}")

    df["tipo"] = df["dx"].apply(lambda x: "Benigna" if x in benignas else "Maligna")
    conteo_tipo = df["tipo"].value_counts().to_dict()
    mostrar_distribucion(conteo_tipo, "HAM10000 - Benignas vs Malignas")

    w, h = analizar_tamanio_promedio(HAM_DIR)
    print(f"📐 Tamaño promedio de imágenes: {w}x{h}px")


# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    analizar_ham10000()
