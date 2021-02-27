from setuptools import setup
with open(__file__ + '/../test-requirements.txt') as f:
    test_requirements = f.read().split('\n')
with open(__file__ + '/../requirements.txt') as f:
    install_requirements = f.read().split('\n')

setup(
    name='tensorguard',
    version='0.1.0',
    packages=['tests', 'shapeguard'],
    url='https://github.com/Michedev/tensorguard',
    license='Apache-2.0',
    author='mikedev',
    author_email='mik3dev@gmail.com',
    description='TensorGuard helps to guard against bad Tensor Shapes',
    test_require=test_requirements,
    install_requires=install_requirements
)
