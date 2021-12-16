from setuptools import setup, find_packages

setup(name='kornia_io',
      version='0.1.0',
      description='Kornia input and output wrapper.',
      author='Kornia.org',
      url='https://github.com/kornia/kornia_io',
      install_requires=[
          'torch',
          'kornia',
          'opencv-python',
          'depthai',
          'visdom',
      ],
      extras_require={
          'dev': [
              'pytest',
              'pytest-flake8',
              'pytest-cov',
              'pytest-mypy',
              'pytest-pydocstyle',
              'mypy',  # TODO: check if we can remove the deps without pytest-*
              'pydocstyle',
              'flake8',
              'pep8-naming'
          ]
      },
      packages=find_packages(),
      )
