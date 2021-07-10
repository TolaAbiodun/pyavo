import setuptools
import re

verstr = 'unknown'
VERSIONFILE = "pyavo/_version.py"
with open(VERSIONFILE, "r")as f:
    verstrline = f.read().strip()
    pattern = re.compile(r"__version__ = ['\"](.*)['\"]")
    mo = pattern.search(verstrline)
if mo:
    verstr = mo.group(1)
    print("Version "+verstr)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open("README.rst", "r") as f:
    desc = f.read()

setuptools.setup(
    name="pyavo",
    version="1.0.0",
    author="Tola Abiodun",
    author_email="tola.abiodun@fluxgateng.com",
    long_description=desc,
    long_description_content_type="text/markdown",
    url="https://github.com/olawaleibrahim/petroeval",
    packages=setuptools.find_packages(),
    description='AVO Analysis in Python',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        'matplotlib', 'pandas', 'numpy', 'las',
        'xarray', 'segyio', 'bruges']
)
