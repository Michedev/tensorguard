from setuptools import setup
from pathlib import Path

ROOT = Path(__file__).parent
with open(ROOT / 'test-requirements.txt') as f:
    test_requirements = f.read().split('\n')
with open(ROOT / 'requirements.txt') as f:
    install_requirements = f.read().split('\n')

setup(
    name='tensorguard',
    version='0.1.0',
    packages=['tests', 'tensorguard'],
    url='https://github.com/Michedev/tensorguard',
    license='Apache-2.0',
    author='mikedev',
    author_email='mik3dev@gmail.com',
    description='TensorGuard helps to guard against bad Tensor Shapes',
    test_require=test_requirements,
    install_requires=install_requirements,
    long_description="TensorGuard helps to guard against bad Tensor shapes in any tensor based library "
                     "(e.g. Numpy, Pytorch, Tensorflow) using an intuitive symbolic-based syntax"
)
