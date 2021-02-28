import os

from setuptools import setup
from pathlib import Path
import distutils.cmd as cmd

ROOT = Path(__file__).parent
with open(ROOT / 'test-requirements.txt') as f:
    test_requirements = f.read().split('\n')
with open(ROOT / 'requirements.txt') as f:
    install_requirements = f.read().split('\n')
with open(ROOT / 'README.md') as f:
    readme = f.read()

class PytestCmd(cmd.Command):

    user_options = []
    description = 'run pytest on this project'

    def initialize_options(self) -> None:
        print('Be sure to install test-requirements.txt before testing')

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        os.system(f'cd {ROOT.absolute()} && pytest')

setup(
    name='tensorguard',
    version='0.1.1',
    packages=['tensorguard'],
    url='https://github.com/Michedev/tensorguard',
    license='Apache-2.0',
    author='mikedev',
    author_email='mik3dev@gmail.com',
    description='TensorGuard helps to guard against bad Tensor Shapes',
    tests_require=test_requirements,
    setup_requires=['pytest-runner'],
    install_requires=install_requirements,
    long_description=readme,
    long_description_content_type='text/markdown',
    python_requires='>=3.6.0',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
    cmdclass={
        'test': PytestCmd
    }
)
