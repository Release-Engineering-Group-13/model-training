import xml.etree.ElementTree as ET


def parse_results(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    testsuite = root.find('testsuite')
    tests = int(testsuite.attrib['tests'])
    errors = int(testsuite.attrib['errors'])
    failures = int(testsuite.attrib['failures'])
    skipped = int(testsuite.attrib['skipped'])
    return tests, errors, failures, skipped


def update_readme_text(readme_file, pylint_results, pytest_results, flake8_results):
    with open(readme_file, 'r') as file:
        readme_content = file.readlines()

    start_marker = '<!-- results-start -->'
    end_marker = '<!-- results-end -->'

    start_index = readme_content.index(start_marker + '\n')
    end_index = readme_content.index(end_marker + '\n')

    new_content = readme_content[:start_index+1]
    new_content.append('\n### Pytest Results\n\n')
    new_content.append(f'- **Tests:** {pylint_results[0]}\n')
    new_content.append(f'- **Errors:** {pylint_results[1]}\n')
    new_content.append(f'- **Failures:** {pylint_results[2]}\n')
    new_content.append(f'- **Skipped:** {pylint_results[3]}\n')
    new_content.append('\n')
    new_content.append('\n### Pylint Results\n\n')
    new_content.append(f'- **Tests:** {pytest_results[0]}\n')
    new_content.append(f'- **Errors:** {pytest_results[1]}\n')
    new_content.append(f'- **Failures:** {pytest_results[2]}\n')
    new_content.append(f'- **Skipped:** {pytest_results[3]}\n')
    new_content.append('\n')
    new_content.append('\n### Flake8 Results\n\n')
    new_content.append(f'- **Tests:** {flake8_results[0]}\n')
    new_content.append(f'- **Errors:** {flake8_results[1]}\n')
    new_content.append(f'- **Failures:** {flake8_results[2]}\n')
    new_content.append(f'- **Skipped:** {flake8_results[3]}\n')
    new_content.append('\n')
    new_content += readme_content[end_index:]

    with open(readme_file, 'w') as file:
        file.writelines(new_content)


def generate_badge_url(name, results):
    # tests = results[0]
    errors = results[1]
    failures = results[2]
    skipped = results[3]
    if errors > 0 or failures > 0:
        status = "failing"
        color = "red"
    elif skipped > 0:
        status = "unstable"
        color = "yellow"
    else:
        status = "passing"
        color = "brightgreen"

    badge_url = f"https://img.shields.io/badge/{name}-{status}-{color}?logo={name}"
    return badge_url


def update_readme_badge(badge_url_list, readme_file):

    with open(readme_file, 'r') as file:
        readme_content = file.readlines()

    start_marker = '<!-- badge-start -->'
    end_marker = '<!-- badge-end -->'

    start_index = readme_content.index(start_marker + '\n')
    end_index = readme_content.index(end_marker + '\n')

    new_content = readme_content[:start_index+1]
    for badge_url in badge_url_list:
        new_content.append(f'![Pytest]({badge_url})\n')
    new_content += readme_content[end_index:]

    with open(readme_file, 'w') as file:
        file.writelines(new_content)


if __name__ == '__main__':
    readme_file = 'README.md'

    pylint_results = parse_results('pylint-report.xml')
    pytest_results = parse_results('pytest-report.xml')
    flake8_results = parse_results('flake8-report.xml')
    update_readme_text(readme_file, pylint_results, pytest_results, flake8_results)

    badge_url_list = []
    badge_url_list.append(generate_badge_url("pylint", pylint_results))
    badge_url_list.append(generate_badge_url("pytest", pytest_results))
    badge_url_list.append(generate_badge_url("flake8", flake8_results))
    update_readme_badge(badge_url_list, readme_file)
