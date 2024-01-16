from setuptools import setup, find_packages

setup(
  name = 'band-split-rope-transformer ',
  packages = find_packages(exclude=[]),
  version = '',
  license='MIT',
  description = '',
  author = '',
  author_email = '',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/kweiwen/band-split-rope-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'music source separation'
  ],
  install_requires=[
    'beartype',
    'einops>=0.6.1',
    'librosa',
    'rotary-embedding-torch>=0.3.6',
    'torch>=2.0',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)