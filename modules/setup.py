from setuptools import setup

setup(
    name='KBMproject',
    version='0.0.1',
    install_requires=[
        'citylearn',
        'torch',
        'adversarial-robustness-toolbox',
        'tqdm',
        'pandas',
        'numpy',
    ],
)