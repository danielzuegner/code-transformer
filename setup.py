from setuptools import setup

setup(name='code_transformer',
      version='0.1',
      description='Code Transformer',
      author='Daniel Zügner, Tobias Kirschstein, Michele Catasta, Jure Leskovec, Stephan Günnemann',
      author_email='zuegnerd@in.tum.de,kirschto@in.tum.de',
      packages=['code_transformer'],
      install_requires=['numpy', 'scipy', 'matplotlib', 'torch', 'tqdm',
                        'jsonlines', 'jsonpickle', 'networkx', 'sacred', 'joblib',
                        'pandas', 'transformers', 'rouge', 'Pygments', 'PyYAML', 'environs'],
      zip_safe=False)
