from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="connect4_robot_j4",
    version="0.1.0",
    author="Vaucanson Robot J4 Team",
    description="Un Puissance 4 robotisé avec vision par ordinateur, IA et Arduino",
    packages=find_packages(), # Automatically find all packages in the directory
    install_requires=requirements,
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "connect4=main:main"
        ]
    },
    include_package_data=True,
)