from setuptools import setup, find_packages

setup(
    name='bayes_opt',
    version='1',
    packages=find_packages(),
    include_package_data = True,
    description='BOIL',
    install_requires=[
        #"numpy >= 1.10.0",
        "numpy",
        "scipy = 1.4.1",
        "scikit-learn >= 1.0.2",
        "tabulate>=0.8.7",
        "matplotlib>=3.1.0",
        "tensorflow>=2.8.0",
        "gym>=0.5",
        "sobol-seq>=0.2.0",
        "tensorflow-probability>=0.16.0",
        "tqdm>=4.64.0",
        "pygame>=2.1.0"
    ],
)
