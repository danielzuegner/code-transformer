from setuptools import setup, find_packages

setup(name='code_transformer',
      version='0.1',
      description='Code Transformer',
      author='Daniel Zügner, Tobias Kirschstein, Michele Catasta, Jure Leskovec, Stephan Günnemann',
      author_email='zuegnerd@in.tum.de,kirschto@in.tum.de',
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'matplotlib', 'torch', 'tqdm',
                        'jsonlines', 'jsonpickle<2.0.0', 'networkx', 'sacred', 'joblib',
                        'pandas', 'transformers', 'rouge', 'Pygments', 'PyYAML', 'environs'],
      zip_safe=False)
