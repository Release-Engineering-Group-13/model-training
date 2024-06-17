import xml.etree.ElementTree as ET

def parse_test_results(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    testsuite = root.find('testsuite')
    tests = int(testsuite.attrib['tests'])
    errors = int(testsuite.attrib['errors'])
    failures = int(testsuite.attrib['failures'])
    skipped = int(testsuite.attrib['skipped'])
    return tests, errors, failures, skipped

def update_readme_text(tests, errors, failures, skipped, readme_file):
    with open(readme_file, 'r') as file:
        readme_content = file.readlines()
    
    start_marker = '<!-- pytest-results-start -->'
    end_marker = '<!-- pytest-results-end -->'
    
    start_index = readme_content.index(start_marker + '\n')
    end_index = readme_content.index(end_marker + '\n')
    
    new_content = readme_content[:start_index+1]
    new_content.append(f'\n### Pytest Results\n\n')
    new_content.append(f'- **Tests:** {tests}\n')
    new_content.append(f'- **Errors:** {errors}\n')
    new_content.append(f'- **Failures:** {failures}\n')
    new_content.append(f'- **Skipped:** {skipped}\n')
    new_content.append(f'\n')
    new_content += readme_content[end_index:]
    
    with open(readme_file, 'w') as file:
        file.writelines(new_content)

def generate_badge_url(tests, errors, failures, skipped):
    if errors > 0 or failures > 0:
        status = "failing"
        color = "red"
    elif skipped > 0:
        status = "unstable"
        color = "yellow"
    else:
        status = "passing"
        color = "brightgreen"

    badge_url = f"https://img.shields.io/badge/tests-{status}-{color}?logo=pytest"
    return badge_url

def update_readme_badge(badge_url, readme_file):
    with open(readme_file, 'r') as file:
        readme_content = file.readlines()
    
    start_marker = '<!-- pytest-badge-start -->'
    end_marker = '<!-- pytest-badge-end -->'
    
    start_index = readme_content.index(start_marker + '\n')
    end_index = readme_content.index(end_marker + '\n')
    
    new_content = readme_content[:start_index+1]
    new_content.append(f'![Pytest]({badge_url})\n')
    new_content += readme_content[end_index:]
    
    with open(readme_file, 'w') as file:
        file.writelines(new_content)

if __name__ == '__main__':
    xml_report = 'report.xml'
    readme_file = 'README.md'
    tests, errors, failures, skipped = parse_test_results(xml_report)
    update_readme_text(tests, errors, failures, skipped, readme_file)
    badge_url = generate_badge_url(tests, errors, failures, skipped)
    update_readme_badge(badge_url, readme_file)