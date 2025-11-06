from setuptools import setup, find_packages

setup(
    name="flowimds",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
    ],
)
