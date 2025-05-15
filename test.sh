argo list -n ci


 argo submit --watch build.yaml --name test-build -n ci


argo submit build.yaml --watch -p dockerfileContent="$(cat dockerfile)" -n ci



argo submit --watch run.yaml --name run -n ci


