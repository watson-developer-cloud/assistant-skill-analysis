from setuptools import setup, find_packages
import re
import unittest

def find_version():
    with open('_version.py', encoding='utf8') as f:
        contents = f.read()
    print(contents)
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
    name='watson-assistant-skill-analysis-tools',
    version=VERSION,

    description='Dialog Skill Analysis Tool for Watson Assistant',
        url='https://github.com/watson-developer-cloud/assistant-dialog-skill-analysis',

    author='Navneet Rao, Haode Qi, Yang Yu and Ming Tan',
    author_email='nrao@us.ibm.com',

    test_suite='src.test.python',

    packages=find_packages(exclude=["src/test"]),

    include_package_data=True,

    install_requires=REQUIREMENTS,

)
