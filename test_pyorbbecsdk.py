#!/usr/bin/env python3
"""Test script to verify pyorbbecsdk is installed correctly."""

import sys


def test_pyorbbecsdk_import():
    """Test that pyorbbecsdk can be imported."""
    try:
        import pyorbbecsdk
        print("pyorbbecsdk imported successfully")
        return True
    except ImportError as e:
        print(f"Failed to import pyorbbecsdk: {e}")
        return False


def test_pyorbbecsdk_version():
    """Test that pyorbbecsdk version can be retrieved."""
    try:
        import pyorbbecsdk
        if hasattr(pyorbbecsdk, "__version__"):
            print(f"pyorbbecsdk version: {pyorbbecsdk.__version__}")
        else:
            print("pyorbbecsdk version attribute not available")
        return True
    except Exception as e:
        print(f"Error getting version: {e}")
        return False


def test_pyorbbecsdk_context():
    """Test that pyorbbecsdk Context can be created."""
    try:
        from pyorbbecsdk import Context
        ctx = Context()
        print("pyorbbecsdk Context created successfully")
        return True
    except Exception as e:
        print(f"Failed to create Context: {e}")
        return False


def test_list_devices():
    """Test listing available devices."""
    try:
        from pyorbbecsdk import Context
        ctx = Context()
        device_list = ctx.query_devices()
        device_count = device_list.get_count()
        print(f"Found {device_count} Orbbec device(s)")
        return True
    except Exception as e:
        print(f"Failed to list devices: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing pyorbbecsdk installation")
    print("=" * 50)

    tests = [
        ("Import test", test_pyorbbecsdk_import),
        ("Version test", test_pyorbbecsdk_version),
        ("Context test", test_pyorbbecsdk_context),
        ("List devices test", test_list_devices),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        result = test_func()
        results.append((name, result))

    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)

    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
