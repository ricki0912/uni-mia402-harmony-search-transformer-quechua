import os
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from search_orchestrator import main as optimize_main
from utils.logger import Logger

def main():
    logger = Logger()
    logger.print("Iniciando programa...")
    print("Iniciando búsqueda de hiperparámetros con Harmony Search...")
    optimize_main()

if __name__ == "__main__":
    main()