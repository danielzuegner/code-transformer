from setuptools import setup

setup(name='code_transformer',
      version='0.1',
      description='Code Transformer',
      author='Daniel Zügner, Tobias Kirschstein, Michele Catasta, Jure Leskovec, Stephan Günnemann',
      author_email='zuegnerd@in.tum.de,kirschto@in.tum.de',
      packages=['code_transformer'],
      install_requires=['jsonlines==1.2.0', 'rouge==1.0.0', 'tensorflow==1.15.0', 'joblib==0.14.1', 
                        'scipy==1.4.1', 'networkx==2.4', 'Pygments==2.6.1', 'torch==1.4.0', 
                        'numpy==1.18.1', 'jsonpickle==1.3', 'pandas==1.0.5', 'tqdm==4.43.0',
                        'transformers==3.1.0', 'six==1.14.0', 'pytest==6.2.2', 'PyYAML==5.4.1',
                        'Requests==2.23.0', 'scikit_learn==0.24.1'],
      zip_safe=False)
