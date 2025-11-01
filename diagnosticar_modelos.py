# diagnosticar_modelos.py corregido
import pickle
import json
import numpy as np
from pathlib import Path

def diagnosticar_modelos():
    # USAR RUTA CORRECTA - desde la raíz del proyecto
    base_models = Path("models")  # Cambiado a "models"
    
    print("=== DIAGNÓSTICO DE MODELOS ===")
    print(f"Buscando en: {base_models.absolute()}")
    
    # Verificar Holt-Winters
    print("\n🔍 Holt-Winters models:")
    for hw_file in ["hw_gasolina_super.pkl", "hw_diesel.pkl"]:
        ruta = base_models / hw_file
        if ruta.exists():
            try:
                file_size = ruta.stat().st_size
                print(f"✅ {hw_file}: {file_size} bytes - EXISTE")
                
                # Intentar cargar
                with open(ruta, "rb") as f:
                    modelo = pickle.load(f)
                print(f"   ✅ Carga exitosa - Tipo: {type(modelo)}")
                
                # Verificar si tiene método de forecast
                if hasattr(modelo, "forecast"):
                    print(f"   ✅ Tiene método forecast")
                else:
                    print(f"   ⚠️  No tiene método forecast")
                    
            except Exception as e:
                print(f"   ❌ Error cargando: {e}")
        else:
            print(f"❌ {hw_file}: NO ENCONTRADO en {ruta.absolute()}")
    
    # Verificar MLP
    print("\n🔍 MLP models:")
    for mlp_file in ["mlp_gasolina_super.pkl", "mlp_diesel.pkl"]:
        ruta = base_models / mlp_file
        if ruta.exists():
            try:
                file_size = ruta.stat().st_size
                print(f"✅ {mlp_file}: {file_size} bytes - EXISTE")
                
                with open(ruta, "rb") as f:
                    modelo = pickle.load(f)
                print(f"   ✅ Carga exitosa - Tipo: {type(modelo)}")
                
            except Exception as e:
                print(f"   ❌ Error cargando: {e}")
        else:
            print(f"❌ {mlp_file}: NO ENCONTRADO en {ruta.absolute()}")
    
    # Verificar Prophet
    print("\n🔍 Prophet models:")
    for prop_file in ["prophet_gasolina_super.json", "prophet_diesel.json"]:
        ruta = base_models / prop_file
        if ruta.exists():
            try:
                file_size = ruta.stat().st_size
                print(f"✅ {prop_file}: {file_size} bytes - EXISTE")
                
                with open(ruta, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"   ✅ Carga JSON exitosa")
                print(f"   Tipo de datos: {type(data)}")
                
                if isinstance(data, dict):
                    print(f"   Keys: {list(data.keys())}")
                    for k, v in data.items():
                        if isinstance(v, list):
                            print(f"   {k}: {len(v)} elementos")
                            if len(v) > 0:
                                print(f"     Primeros 3: {v[:3]}")
                elif isinstance(data, list):
                    print(f"   Lista con {len(data)} elementos")
                    if len(data) > 0 and isinstance(data[0], dict):
                        print(f"   Columnas del primer elemento: {list(data[0].keys())}")
                        
            except Exception as e:
                print(f"   ❌ Error cargando JSON: {e}")
        else:
            print(f"❌ {prop_file}: NO ENCONTRADO en {ruta.absolute()}")

if __name__ == "__main__":
    diagnosticar_modelos()