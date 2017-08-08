from setuptools import setup

setup(name='yearonequant',
      version='1.5.4',
      description='Think as quant, trade as quant',
      url='http://github.com/hyqLeonardo/yearonequant',
      author='Leonardo',
      author_email='hyq335335@163.com',
      license='MIT',
      packages=['yearonequant'],
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'plotly',
          'rqdatac'
      ],
      zip_safe=False)
