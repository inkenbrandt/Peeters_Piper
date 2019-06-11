from setuptools import setup


setup(
    name="peeter_piper",
    description="generate piper diagram used in water chemistry analysis; from Peeters 2014"
    py_modules=["peeter_piper"],
    version="0.1.1",
    url="https://github.com/inkenbrandt/peeter_piper",
    install_requires=["numpy", "matplotlib", "scipy"],
)
