from setuptools import setup


setup(name='lipopt',
      version='0.1',
      description='Lipschitz Constant Estimation of Neural Networks',
      author='Fabian Latorre',
      author_email='latorrefabian@gmail.com',
      license='MIT',
      packages=['lipopt'],
      install_requires=[
          'torch', 'pytest'
      ],
      zip_safe=False)

