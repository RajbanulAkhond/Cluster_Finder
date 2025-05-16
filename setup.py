"""
Setup configuration for the cluster_finder package.
"""
import sys
import platform
import subprocess
import logging
from setuptools import setup, find_packages

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger('setup')

def check_lzma_support():
    """Check if LZMA compression is supported in this Python installation."""
    try:
        import lzma
        return True
    except ImportError:
        return False

def print_lzma_installation_instructions():
    """Print instructions for installing LZMA support."""
    system = platform.system()
    logger.error("Python LZMA support is missing!")
    logger.error("This is required by dependencies like pymatgen for compression features.\n")
    
    logger.info("To fix this issue:")
    if system == "Darwin":  # macOS
        logger.info("1. Install xz: brew install xz")
    elif system == "Linux":
        # Try to detect the Linux distribution
        try:
            import distro
            linux_distro = distro.id()
        except ImportError:
            try:
                with open('/etc/os-release') as f:
                    for line in f:
                        if line.startswith('ID='):
                            linux_distro = line.split('=')[1].strip().strip('"')
                            break
            except:
                linux_distro = ""
        
        if linux_distro.lower() in ['ubuntu', 'debian']:
            logger.info("1. Install liblzma-dev: sudo apt-get install liblzma-dev")
        elif linux_distro.lower() in ['centos', 'rhel', 'fedora']:
            logger.info("1. Install xz-devel: sudo yum install xz-devel")
        else:
            logger.info("1. Install the LZMA development package for your distribution")
    elif system == "Windows":
        logger.info("1. Reinstall Python using the official installer from python.org")
        logger.info("   Make sure to select the option to install all optional features")
    
    # Check if using pyenv
    if 'pyenv' in sys.executable:
        python_version = ".".join(map(str, sys.version_info[:3]))
        if system == "Darwin":  # macOS
            logger.info("\nFor pyenv users:")
            logger.info(f"  LDFLAGS=\"-L$(brew --prefix xz)/lib\" CPPFLAGS=\"-I$(brew --prefix xz)/include\" pyenv install {python_version}")
        else:
            logger.info("\nFor pyenv users: Reinstall Python with LZMA support")
            logger.info(f"  pyenv install {python_version}")
    
    logger.info("\n2. After installing the system requirements, reinstall cluster-finder")

# Check for LZMA support
if not check_lzma_support():
    print_lzma_installation_instructions()
    # Continue installation anyway, but warn the user
    logger.warning("Installation will continue, but you may encounter errors when running the package.")
    logger.warning("It's strongly recommended to fix the LZMA issue before using cluster-finder.\n")

# Get long description from README.md if it exists
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A package for finding and analyzing atomic clusters in crystal structures"

setup(
    name="cluster_finder",
    version="0.1.0",
    author="Md. Rajbanul Akhond",
    author_email="makhond@iu.edu",
    description="A package for finding and analyzing atomic clusters in crystal structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RajbanulAkhond/Cluster_Finder",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11.11",
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.4",
        "pymatgen>=2023.0",
        "networkx>=2.6",
        "pandas>=1.3",
        "mp-api>=0.33.3",
        "pydantic>=1.10.8,<3.0.0",
        "ase==3.22.1",
        "aiohttp>=3.8.0,<4.0.0",
        "asyncio>=3.4.3,<4.0.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "joblib>=1.3.0,<2.0.0",
        "pyyaml>=6.0.0,<7.0.0",
        "scipy>=1.7",
        "tqdm>=4.65.0",
        "distro>=1.8.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0,<9.0.0",
            "pytest-cov>=2.12.0",
            "black>=22.1.0",
            "isort>=5.10.0",
            "pylint>=2.12.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "cluster-finder=cluster_finder.cli:app"
        ]
    }
)