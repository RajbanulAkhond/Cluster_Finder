from setuptools import setup, find_packages

setup(
    name="cluster_finder",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.0",
        "pandas==2.0.3",
        "networkx==3.1",
        "pymatgen==2023.8.10",
        "matplotlib==3.7.2",
        "mp-api==0.30.4",
        "pytest==7.4.0",
        "pydantic==1.10.8"
    ],
    entry_points={
        'console_scripts': [
            'cluster-finder=cluster_finder.cli:main',
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
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research"
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black",
            "isort",
            "pylint"
        ]
    }
)