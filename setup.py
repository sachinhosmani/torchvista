from setuptools import setup, find_packages

setup(
    name='torchvista',
    version='0.2.8',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'ipython>=7.0.0',
        'numpy>=1.18.0'
    ],
    python_requires='>=3.8',
    package_data={
        'torchvista': ['templates/*.html', 'assets/*'],
    },
    description="Interactive PyTorch model visualizer for notebooks",
    long_description=(
        "Torchvista displays an interactive graph of the forward pass of a PyTorch model directly in the notebook with a single line of code. "
        "It offers error-tolerant partial visualization to trace the source of errors like tensor shape mismatches"
    ),
    long_description_content_type='text/plain',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",        
    ]
)
