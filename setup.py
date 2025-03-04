from setuptools import setup, find_packages

setup(
    name="cluster_finder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "pymatgen>=2022.0.0",
        "matplotlib>=3.4.0",
        "mp-api>=0.30.0",
        "pytest>=6.0.0"
    ],
    entry_points={
        'console_scripts': [
            'cluster_finder=cluster_finder.cli:main',
        ],
    },
    author="Md. Rajbanul Akhond",
    author_email="mdakhond@iu.edu",
    description="A package for finding and analyzing clusters in crystal structures",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RajbanulAkhond/Cluster_Finder/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
) 