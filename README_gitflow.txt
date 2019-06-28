#####################################################################

Github Link
https://github.com/Lee-Gihun/Micronet_GSJ

#####################################################################

Gitflow Link
https://danielkummer.github.io/git-flow-cheatsheet/index.ko_KR.html

####################################################################

Useful Command for git

git branch -a : show all branch

git checkout <branch_name> : move to that branch_name

git flow feature start <branch_name> : make feature/branch_name

git flow feature publish <branch_name> : show feature/branch_name on github

git flow feature finish <branch_name> : merge to develop branch(when you make branch from develop)

git status : show the status of current repo (which file or directory can be added or committed)

git log : can see some logs of what you did

git config user.name "name" : change config user name

git config user.email "email@kaist.ac.kr" : change config user email

git add <file_name> : make some file

git rm <file_name> : remove some file

git commit -m "commit message" : commit the added file of removed file with some message 

git commit --amend --author='user.name<some_email@kaist.ac.kr>'

git reset --hard HEAD~1 : cancel the last commit (BE CAREFUL TO USE THIS)

git push : push to remote branch

git pull : pull some changes from remote branch to local branch

git merge develop : when you use this command in master, you can pull changes from develop branch

git stash : store the last add(change)

git stash apply : apply the stored change

######################################################################

Working order

1. git checkout develop
cf) If develop branch do not follow master, use <git pull origin master> command

2. git flow feature start <branch_name>

3. Do some works!

4. git add <3 results>

5. git commit -m "3 works message"

6. git flow feature publish <branch_name>

# other people say it's OK!
7. git checkout develop

8. git flow feature finish <branch_name>

9. git checkout master

10. git merge develop

11. check the comment git suggest... (git pull origin master & git push 
				or show failed message - automatically merged failed and conflicted..
				if then solve conflicted part of file and add, commit, push!)

#######################################################################
