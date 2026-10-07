import sys


def test_imports():
    import flowerlite
    assert flowerlite.__version__ == "0.1.0"

def test_python_version():
    assert sys.version_info >= (3, 10)
    assert sys.version_info < (3, 12)
