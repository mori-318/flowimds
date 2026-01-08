from setuptools import setup, find_packages

setup(
    name="flowimds",
    version="1.0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
    ],
)
