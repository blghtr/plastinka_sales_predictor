import pytest
import os

def test_fs_fixture_creates_file(fs):
    # fs is the pyfakefs fixture
    fs.create_file("/fake_dir/fake_file.txt", contents="Hello Kilo Code!")
    assert os.path.exists("/fake_dir/fake_file.txt")
    with open("/fake_dir/fake_file.txt", "r") as f:
        content = f.read()
    assert content == "Hello Kilo Code!"

def test_fs_fixture_creates_directory(fs):
    fs.create_dir("/another_fake_dir")
    assert os.path.exists("/another_fake_dir")
    assert os.path.isdir("/another_fake_dir")