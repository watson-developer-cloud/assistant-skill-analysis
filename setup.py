from setuptools import setup, find_packages
import re


def find_version():
    with open('_version.py', encoding='utf8') as f:
        contents = f.read()
    match_result = re.search(r'^__version__ *= *\'(.*?)\'', contents, re.MULTILINE)
    if match_result:
        return match_result.group(1)
    raise Exception('Unable to find version string')


def get_requirements():
    with open('requirements.txt', encoding='utf8') as f:
        requirements = f.read().splitlines()
    return requirements


VERSION = find_version()

REQUIREMENTS = get_requirements()

setup(
    name='assistant_dialog_skill_analysis',
    version=VERSION,
    author='Haode Qi, Navneet Rao, Ming Tan, Yang Yu, Panos Karagiannis and Ladislav Kunc',
    author_email='nrao@us.ibm.com',
    description='Dialog Skill Analysis Tool for Watson Assistant',
    url='https://github.com/watson-developer-cloud/assistant-dialog-skill-analysis',
    license='Apache License 2.0',
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=['tests']),
    test_suite='tests',
    include_package_data=True,
)
