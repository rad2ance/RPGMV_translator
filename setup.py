from setuptools import setup, find_packages

setup(
    name='RPGMV_translator',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'openai>=1.0.0',
        'tqdm>=4.0.0',
        'tiktoken>=0.7.0',
    ],
    entry_points={
        'console_scripts': [
            'mvtrans=rpgmv_translator.main:main',
        ],
    },
    # Add other metadata like author, description, etc.
)
