#!/usr/bin/python
# pre-commit hook
# ensure that unit tests are ok if you push to develop or master
# @author Stefan Hegglin, Adrian Oeftiger

import subprocess as sbp
import sys

def run():
    '''Run all tests for PyHEADTAIL here. Return success status.'''
    return test_all() == 0

def test_all():
    # get the git root directory
    git_dir = sbp.check_output(
        ["git", "rev-parse", "--show-toplevel"]).rstrip().decode("utf-8")
    # construct the testsuite filename
    fn_testsuite = git_dir + '/PyHEADTAIL/testing/unittests/testsuite.py'

    # don't test code which isnt part of the commit
    sbp.call(["git", "stash", "-q", "--keep-index"])
    try:
        # run the testsuite (exit status 0: all fine)
        res = sbp.call(["python", fn_testsuite])
    finally:
        # pop the stash
        sbp.call(["git" , "stash", "pop", "-q"])

    return res

if __name__ == '__main__':
    # get the current branch name, strip trailing whitespaces using rstrip()
    branch = sbp.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]).rstrip()

    if branch == 'master' or branch == 'develop':
        print ('\n' + 'X' * 66)
        print ('You are trying to push to the master or develop branch.')
        print ('Checking unit tests first...')
        print ('X' * 66 + '\n')

        res = test_all()
        if res == 0:
            print ('\n' + 'X' * 66)
            print ('Passed unit tests, proceeding.')
            print ('X' * 66 + '\n')
        else:
            print ('\n' + 'X' * 66)
            print ('Failed unit tests, fix them before comitting. Aborting.')
            print ('X' * 66 + '\n')
        sys.exit(res)
    else:
        # if not on develop or master: continue
        sys.exit(0)
