import re


def grep(pattern, text, values=False, ignore_case=False):
    # Compile the regex pattern with the ignore_case flag if enabled
    flags = re.IGNORECASE if ignore_case else 0
    regex = re.compile(pattern, flags)

    if values:
        # Return the matching lines
        return [line for line in text if regex.search(line)]
    else:
        # Return a list of booleans indicating matches
        return [bool(regex.search(line)) for line in text]
