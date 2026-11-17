
import pytest

from src.transformations.delta_pipeline import create_spark


@pytest.fixture(scope="session")
def spark():
    """Provide a Delta-enabled Spark session for tests."""
    session = create_spark()

    yield session

    session.stop()
