import sys
import subprocess


STATUS_FILE = "properties.json"


packages = ("PyQt5", "texify", "torch", "Pillow")

def install_packages():
    """Install required packages."""
    for package in packages:
        sys.stdout.write(f"Installing {package}...\n")
        subprocess.run([sys.executable, "-m", "pip", "install", package])

install_packages()