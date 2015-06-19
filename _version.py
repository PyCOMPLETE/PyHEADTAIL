import os, subprocess
worktree = os.path.dirname(os.path.abspath(__file__))
gitdir = worktree + '/.git/'
try:
    __version__ = subprocess.check_output(
        'git --git-dir=' + gitdir + ' --work-tree=' +
        worktree + ' describe --long --dirty --abbrev=10 --tags', shell=True)
    __version__ = __version__.rstrip() # remove trailing \n
    __version__ = __version__[1:] # remove leading v
except:
    __version__ = '(no git available to determine version)'
