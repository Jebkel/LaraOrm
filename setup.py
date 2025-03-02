from setuptools import setup, find_packages


def readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


setup(
    name='laraMemeOrm',
    version='0.0.1',
    author='Jebkel',
    author_email='csv666devil@gmail.com',
    description='This is the simplest module for quick work with files.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/Jebkel',
    packages=find_packages(),
    install_requires=["pydantic>=2.10.6", "aiomysql>=0.2.0", "loguru>=0.7.3", "inflection>=0.5.1"],
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords='',
    project_urls={
        'GitHub': 'https://github.com/Jebkel'
    },
    python_requires='>=3.9'
)