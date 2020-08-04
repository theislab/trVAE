from pathlib import Path

from setuptools import setup, find_packages

long_description = Path('README.md').read_text('utf-8')

try:
    from trvae import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''

setup(name='trVAE',
      version='1.1.2',
      description='Condition out-of-sample prediction',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/theislab/trvae',
      author=__author__,
      author_email=__email__,
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
          l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
      ],
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          "License :: OSI Approved :: MIT License",
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
      ],
      doc=[
          'typing_extensions; python_version < "3.8"',
      ],
      )
