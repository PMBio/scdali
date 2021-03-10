import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dali",
    version="0.0.1",
    author="Tobi Heinen",
    author_email="t.heinen@dkfz.de",
    description="Test for allelic imbalance in single cells.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tohein/dali",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "scipy",
        "chiscore",
        "gpflow",
        "joblib",
        "statsmodels",
        "tqdm"
    ]
)
