#!/usr/bin/python
# pre-commit hook
# ensure that unit tests are ok if you push to develop or master
# @author Stefan Hegglin

import subprocess as sbp
import sys


# get the current branch name, strip trailing whitespaces using rstrip()
branch = sbp.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).rstrip()
# get the git root directory
git_dir = sbp.check_output(["git", "rev-parse", "--show-toplevel"]).rstrip()
# construct the testsuite filename
fn_testsuite = git_dir + '/testing/unittests/testsuite.py'
if branch == 'master' or branch == 'develop':
    print('')
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print('You are trying to push to the master or develop branch.')
    print('Checking unit tests first...')
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print('')

    # don't test code which isnt part of the commit
    sbp.call(["git", "stash", "-q", "--keep-index"])
    # run the testsuite
    res = sbp.call(["python", fn_testsuite])
    # pop the stash
    sbp.call(["git" , "stash", "pop", "-q"])
    if res == 0:
        print('')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' +
                'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print 'Passed unit tests, proceeding.'
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' +
                'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    else:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' +
                'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print 'Failed unit tests, fix them before comitting. Aborting.'
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' +
                'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    sys.exit(res)
else:
    # if not on develop or master: continue
    sys.exit(0)
