from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.readlines()[1]

setup(name='trVAE',
      version='0.0.1',
      description='a deep generative model which learns mapping between multiple different styles',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/theislab/trVAE',
      author='Mohsen Naghipourfar, Mohammad Lotfollahi',
      author_email='mohsen.naghipourfar@gmail.com, mo.lotfollahi@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      )