from setuptools import setup

setup(
    name='loman',
    version='0.1.2',
    packages=['loman'],
    url='https://github.com/janusassetallocation/loman',
    license='BSD',
    author='Ed Parcell',
    author_email='edparcell@gmail.com',
    description='Loman tracks state of computations, and the dependencies between them, allowing full and partial recalculations.',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=['six', 'dill', 'graphviz', 'networkx', 'pandas'],
)
