from setuptools import setup, find_packages

setup(
    name='multi_agent_reach_avoid_game',
    version='0.1',
    packages=find_packages(),  # Automatically finds 'MARAG' and submodules
    install_requires=[
        # Add any dependencies here, e.g., 'numpy', 'matplotlib', etc.
    ],
    author='Hanyang',
    author_email='huhy97@outlook.com',
    description='Multi-Agent Reach-Avoid Game simulation and planning framework.',
    python_requires='>=3.7',
)
