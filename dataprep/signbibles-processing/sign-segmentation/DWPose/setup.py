from setuptools import setup, find_namespace_packages

setup(
    name="dwpose",
    version="0.1",
    packages=find_namespace_packages(include=["*"]),
    install_requires=[
        "numpy",
        "opencv-python",
        "onnxruntime-gpu",
        "einops"
    ]
)
