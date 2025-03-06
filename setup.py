from setuptools import setup, find_packages

setup(
    name="trust_ai",
    version="0.1.0",
    description="A Machine Learning Library for Trustworthy AI model Generation and Evaluation",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "shap",
        "matplotlib",
        "aif360",
        "interpret",
        "BlackBoxAuditing"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
