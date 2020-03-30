import re


def filenameToLabel(file_name):
    """
    Args:
        file_name (string)

    Returns:
        string | None
    """
    result = re.search("^\d+-\d+-(\w+).png$", file_name)
    return result[1] if result else None


def isPng(file_name):
    return file_name[-4:].lower() == ".png"
