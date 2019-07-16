from setuptools import setup

setup(
    name="ddc_pub",
    version="0.1",
    description="Neural network to encode/decode molecules.",
    url="https://bitbucket.astrazeneca.net/users/kjmv588/repos/ddc_pub",
    author="Panagiotis-Christos Kotsias",
    author_email="panagiotis-christos.kotsias@astrazeneca.net",
    license="GPLv3",
    packages=["ddc_pub"],
    install_requires=[
        "numpy",
        "h5py",
        "ipywidgets",
        "tensorflow-gpu==1.12.0",
        "keras",
        "scikit-learn",
        "scipy",
        "ipykernel",
        "ipython",
        "matplotlib",
        "pandas",
    ],
    zip_safe=False,
)
