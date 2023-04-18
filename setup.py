from setuptools import setup

setup(
    name="yim_dataset",
    version="0.1",
    url="https://github.com/ChristophReich1996/Yeast-in-Microstructures-Dataset",
    license="MIT License",
    author="Christoph Reich",
    author_email="ChristophReich@gmx.net",
    description="Code of Yeast in Microstructures Dataset.",
    packages=[
        "yim_dataset",
        "yim_dataset.data",
        "yim_dataset.eval",
        "yim_dataset.vis",
    ],
    install_requires=[
        "torch>=1.0.0",
        "numpy",
        "matplotlib",
    ],
)
