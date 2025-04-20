from setuptools import setup, find_packages

setup(
    name='wildfire',
    version='0.1',
    packages=find_packages(include=['wildfire', 'wildfire.*']),
    install_requires=[
        'tensorflow',
        'wandb',
        'numpy',
        'pandas',
        'matplotlib',
        # add more if needed
    ],
    python_requires='>=3.8',
)