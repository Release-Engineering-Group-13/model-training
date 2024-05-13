"""This module uploads the model to a google drive."""

import ellipsis as el


def main():
    token = el.account.logIn('REMLA13', 'REMLAEllipsis')

    paths = el.path.search(text = "model.joblib", token = token, root=['myDrive'])['result']
    if len(paths):
        el.path.trash(paths[0]['id'], token)
        
    pathId = el.path.file.add(filePath = "data/interim/model.joblib",  token = token)['id']
    el.path.editPublicAccess(pathId = pathId, token = token, access={"accessLevel": 100})

    with open("output/model_id", "w") as text_file:
        text_file.write(pathId)


if __name__ == "__main__":
    main()

