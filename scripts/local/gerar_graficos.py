"""Entrada local para renderizar e validar os gráficos do projeto."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.render import main

if __name__ == '__main__':
    main()
