import re

files = ['w3_practice.qmd', 'w4_practice.qmd', 'w5_practice.qmd', 'w6_practice.qmd']

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all {python} with {python} followed by the options
    # But only if the options are not already there
    pattern = r'```{python}(\n)(?!#\| message: false)'
    replacement = r'```{python}\1#| message: false\n#| warning: false'

    new_content = re.sub(pattern, replacement, content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f'Updated {file_path}')
