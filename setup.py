from setuptools import setup, find_packages

setup(
    name="scMoFuse",
    version="1.0.0",
    description="Single-cell Multi-omics Fusion with Adaptive Cross-attention",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "numpy>=1.23",
        "scipy>=1.10",
        "scikit-learn>=1.2",
    ],
    extras_require={
        "data": ["anndata>=0.8", "scanpy>=1.9"],
        "viz": ["matplotlib>=3.6", "tqdm>=4.65"],
    },
)
