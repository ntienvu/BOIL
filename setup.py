from setuptools import setup, find_packages

setup(
    name='bayes_opt',
    version='1',
    packages=find_packages(),
    include_package_data = True,
    description='BOIL',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.16.1",
        "tabulate>=0.8.7",
        "matplotlib>=3.1.0",
        "tensorflow>=2.0",
        "gym>=0.5"
    ],
)
