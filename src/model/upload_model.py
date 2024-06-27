"""This module uploads the model to ellipsis."""

import ellipsis as el


def main():
    '''Uploads model to ellipsis'''
    token = el.account.logIn('REMLA13', 'REMLAEllipsis')

    paths = el.path.search(text="model.joblib", token=token, root=['myDrive'])['result']
    if len(paths):
        el.path.trash(paths[0]['id'], token)

    path_id = el.path.file.add(filePath="data/interim/model.joblib",
                               parentId="12b2838a-fa7e-4215-bb50-069df2879311", token=token)['id']
    el.path.editPublicAccess(pathId=path_id, token=token, access={"accessLevel": 100})

    with open("output/model_id", "w", encoding="utf-8") as text_file:
        text_file.write(path_id)


if __name__ == "__main__":
    main()
