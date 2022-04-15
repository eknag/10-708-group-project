git config core.hooksPath .githooks/
chmod u+x .githooks/pre-push
chmod u+x .githooks/post-merge
conda env create -f env.yml
conda activate 10-708-group-project