from setuptools import setup

setup(
    name='loman',
    version='0.4.1',
    packages=['loman'],
    url='https://github.com/janusassetallocation/loman',
    license='BSD',
    author='Ed Parcell',
    author_email='edparcell@gmail.com',
    description='Loman tracks state of computations, and the dependencies between them, allowing full and partial recalculations.',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    install_requires=['decorator', 'dill', 'pydotplus', 'networkx', 'pandas', 'matplotlib'],
    extras_require={
        'test': ['pytest'],
    },
)
