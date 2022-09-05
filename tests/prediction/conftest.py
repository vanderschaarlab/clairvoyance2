# Mark all tests inside this directory as `model` tests.
def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker("model")
