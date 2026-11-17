
import pytest

from src.transformations.delta_pipeline import create_spark


@pytest.fixture(scope="function")
def spark():
    """
    Provide a Delta-enabled Spark session
    for each test function.
    """
    session = create_spark()

    try:
        yield session
    finally:
        session.stop()
