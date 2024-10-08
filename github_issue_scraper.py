import pandas as pd
import requests
from datetime import datetime
import sys

# enter token here
token = ''

per_page = 100
# all cFS repos Daniel had
repos = ['CCDD', 'CF', 'cFE', 'cFS', 'ci_lab', 'CS', 'DS', 'ECI', 'elf2cfetbl', 'FM', 'fs_lib',
         'gen_msgids', 'gen_sch_tbl', 'HK', 'HS', 'LC', 'MD', 'MM', 'osal', 'PSP', 'sample_app',
         'sample_lib', 'SC', 'SCA', 'sch_lab', 'SCH', 'tblCRCTool', 'to_lab']
# F' repo
# repos = ['fprime']
for repo in repos:
    page = 1
    print(f'Scraping issues from {repo}...')
    loop = True
    all_issues = []
    filtered_issues = []
    # gather issues in pages (access limit)
    while loop:
        r = requests.get(url=f'https://api.github.com/repos/nasa/{repo}/issues',
                         headers={'Authorization': f'token {token}'},
                         params={'page': page, 'per_page': per_page, 'state': 'all'})
        pass
        if r.status_code == 200:
            issues = r.json()
            pass
            all_issues.extend(issues)
            if len(issues) < per_page:
                loop = False
            page += 1
        else:
            print(f'Repository not found, code {r.status_code}.')
            loop = False
    print(f'Successfully scraped {page - 1} page(s) from {repo}.')
    accepted_labels = ['bug', 'enhancement', 'documentation', 'docs', 'feature', 'question']
    # for progress bar
    counter = 1
    for issue in all_issues:
        # progress bar stuff
        print(f'\rProcessing issues, {round(counter/len(all_issues)*100, 2)}% complete.', end='')
        sys.stdout.flush()
        counter += 1
        # remove pull requests
        if 'pull_request' not in issue:
            # check if a label exists on the issue
            if issue['labels']:
                # get all current labels on the issue
                label_names = []
                for label in issue['labels']:
                    label_names.append(label['name'])
                # make sure the issue has one of the accepted labels
                label_names = [l for l in label_names if l in accepted_labels]

                if label_names:
                    # get all events on the issue
                    issue['events'] = []
                    r = requests.get(issue['events_url'],
                                     headers={'Authorization': f'token {token}'})
                    if r.status_code == 200:
                        events = r.json()

                        # save events regarding labels only
                        for event in events:
                            if event['event'] in ['labeled']:
                                if event['label']['name'] in label_names:
                                    issue_created_at = datetime.strptime(issue['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                                    label_created_at = datetime.strptime(event['created_at'], "%Y-%m-%dT%H:%M:%SZ")
                                    # omitted body since markdown tables just broke everything for some reason
                                    try:
                                        temp_dict = {'title': issue['title'],
                                                     'url': issue['html_url'],
                                                     'label': event['label']['name'],
                                                     'originator': issue['user']['login'],
                                                     'tagged_by': event['actor']['login'],
                                                     'same_actor': (issue['user']['login'] == event['actor']['login']),
                                                     'label_delay_min': round((label_created_at - issue_created_at).total_seconds() / 60.0, 2)}
                                        filtered_issues.append(temp_dict)
                                    finally:
                                        break

    if (len(filtered_issues)) > 0:
        filtered_df = pd.DataFrame(filtered_issues)
        filtered_df.to_csv(f'{repo}_label_info.csv', index=False)
        print(f'\n{len(filtered_issues) + 1} {repo} issue(s) saved to {repo}_label_info.csv.')
    else:
        print(f'\nNo issues in {repo} fit the criteria.')
    print('\n')