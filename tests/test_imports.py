import pytest


@pytest.fixture(scope="session", autouse=True)
def session_fixture():
    # Setup code: Initialize resources needed for the entire test session
    print("Setting up resources for the test session.", end="")
    print("...Done")

    yield

    # Teardown code: Clean up resources after all tests have run
    from bears.util.concurrency._asyncio import _cleanup_event_loop

    print("Tearing down resources after the test session.", end="")
    _cleanup_event_loop()
    print("...Done")


def test_import_main_module():
    import synthesizrr

    assert synthesizrr._LIBRARY_NAME == "synthesizrr"
