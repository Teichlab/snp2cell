from setuptools import setup, find_packages

setup(
    name="snp2cell",
    version="0.3.0",
    description="A package for finding enriched regulatory networks from GWAS and single cell data",
    url="https://github.com/Teichlab/snp2cell",
    author="J.P.Pett",
    author_email="jp30@sanger.ac.uk",
    license="BSD 3-clause",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["snp2cell=snp2cell.cli:app"],
    },
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scanpy",
        "networkx",
        "pyranges",
        "pybiomart",
        "statsmodels",
        "joblib",
        "tqdm",
        "matplotlib",
        "seaborn",
        "typer",
        "typing_extensions",
        "rich",
        "dill",
    ],
    tests_require=[
        "pytest",
    ],
    python_requires=">=3.5, <3.12",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
    ],
)
