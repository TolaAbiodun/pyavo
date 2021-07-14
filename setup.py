# -*- coding: utf-8 -*-
"""
Python installation file.
"""
import setuptools
import re

verstr = 'unknown'
VERSIONFILE = "pyavo/_version.py"
with open(VERSIONFILE, "r") as f:
    verstrline = f.read().strip()
    pattern = re.compile(r"__version__ = ['\"](.*)['\"]")
    mo = pattern.search(verstrline)
if mo:
    verstr = mo.group(1)
    print("Version " + verstr)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setuptools.setup(
    name="pyavo",
    version=verstr,
    author="Tola Abiodun",
    author_email="tola.abiodun@fluxgateng.com",
    long_description=open('README.rst').read(),
    url="https://github.com/TolaAbiodun/pyavo",
    packages=setuptools.find_packages(),
    description='AVO Analysis in Python',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    licence='MIT',
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'matplotlib', 'pandas', 'numpy', 'las',
        'xarray', 'segyio', 'bruges']
)
