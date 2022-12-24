from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f]

setup(
    name='src',
    version='0.1.0',
    description='Multi-label classification with GCN and its variants',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="thanhtvt",
    url='https://github.com/thanhtvt/ml-gcn',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.7',
)
