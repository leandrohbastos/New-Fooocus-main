import os
import subprocess
import sys

def run_command(command):
    print(f"Executando: {command}")
    subprocess.run(command, shell=True, check=True)

def main():
    # Instalar dependências
    print("Instalando dependências...")
    run_command("pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt190/download.html")
    run_command("pip install rembg")
    run_command("pip install -q torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 torchtext==0.16.0 xformers==0.0.22.post7 opencv-python==4.8.1.78 numpy==1.24.3 Pillow==10.0.0 gradio==3.50.2 einops==0.7.0 transformers==4.35.2 accelerate==0.24.1 safetensors==0.4.0 scipy==1.11.3 tqdm==4.66.1 psutil==5.9.6 pytorch_lightning==2.1.2 omegaconf==2.3.0")

    # Limpar e clonar repositório
    print("Clonando repositório...")
    run_command("rm -rf New-Fooocus-main")
    run_command("git clone https://github.com/leandrohbastos/New-Fooocus-main.git")
    os.chdir("New-Fooocus-main")

    # Verificar GPU
    print("Verificando GPU...")
    run_command("nvidia-smi")

    # Iniciar Fooocus
    print("Iniciando Fooocus...")
    run_command("python entry_with_update.py --preset hyper_realistic --share --always-high-vram")

if __name__ == "__main__":
    main() 