from setuptools import setup, find_packages

setup(
    name="quant_lab",
    version="0.1.1",       
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "akshare",
        "matplotlib",    
        "scikit-learn",  
        "lightgbm" ,      
    ],
)