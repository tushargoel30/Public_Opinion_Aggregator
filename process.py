import re
def processSearch(text):
    processed_input = text.strip()

    # Convert to lower case to maintain consistency
    processed_input = processed_input.lower()

    # Optionally remove special characters except spaces
    
    processed_input = re.sub(r'[^a-z0-9 ]', '', processed_input)

    return processed_input