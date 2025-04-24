from setuptools import setup, find_packages

setup(
    name='rollingcv',
    version='0.1.0',
    description='Fixed-window cross-validation for time series (scikit-learn compatible)',
    author='Mariano Tir',
    author_email='marianotir98@gmail.com',
    url='https://github.com/marianotir/rollingcv',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        "numpy>=1.18.0,<2.0",
        "scikit-learn>=0.24"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
