'''
Algorithm to release PyHEADTAIL versions. The structure follows the development
workflow, cf. the PyHEADTAIL wiki:
https://github.com/PyCOMPLETE/PyHEADTAIL/wiki/Our-Development-Workflow

Requires git, hub (the github tool, https://hub.github.com/) and importlib
(included in python >=2.7) installed.

Should be PEP440 conformal.

@copyright: CERN
@date: 26.01.2017
@author: Adrian Oeftiger
'''

import argparse
import importlib # available from PyPI for Python <2.7
import os, subprocess

# CONFIG
version_location = '_version' # in python relative module notation
# (e.g. PyHEADTAIL._version for PyHEADTAIL/_version.py)
test_script_location = 'pre-push' # in python relative module notation
release_branch_prefix = 'release/v' # prepended name of release branch


parser = argparse.ArgumentParser(
    description=(
        'Release a new version of PyHEADTAIL in 2 steps:\n'
        '1. prepare new release from develop branch '
        '(requires release type argument)\n'
        '2. publish new release (requires to be on release branch already)'),
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    'part', type=str,
    choices=['major', 'minor', 'patch'],
    help="release type\n"
         "(use 'minor' for new features "
         "and 'patch' for bugfixes, "
         "'major' is not usually used ;-)",
)

def bumpversion(version, part):
    '''Advance the three component version with format X.Y.Z by one in the
    given part. Return the bumped version string.
    Arguments:
        - version: string, format 'major.minor.patch' (e.g. '1.12.2')
        - part: string, one of ['major', 'minor', 'patch']
    '''
    # exactly 3 components in version:
    parts = version.split('.')
    # only integers stored:
    assert all(p.isdigit() for p in parts)
    major, minor, patch = map(int, parts)
    if part == 'major':
        version = '{0}.0.0'.format(major + 1)
    elif part == 'minor':
        version = '{0}.{1}.0'.format(major, minor + 1)
    elif part == 'patch':
        version = '{0}.{1}.{2}'.format(major, minor, patch + 1)
    else:
        raise ValueError('The given part "' + part + '" is not in '
                         "['major', 'minor', 'patch'].")
    return version

def get_version(version_location):
    '''Retrieve the version from version_location file.'''
    return importlib.import_module(version_location).__version__

def which_part_increases(last_version, new_version):
    '''Return a string which version part is increased. Raise an error
    if new_version is not a valid direct successor to last_version.
    Args:
        - last_version, new_version: string, format 'major.minor.patch'
          (e.g. '1.12.2')
    Return:
        - part: string, one of ['major', 'minor', 'patch']
    '''
    # exactly 3 components in version:
    last_parts = last_version.split('.')
    new_parts = new_version.split('.')
    # only integers stored:
    assert all(p.isdigit() for p in last_parts + new_parts)
    lmajor, lminor, lpatch = map(int, last_parts)
    nmajor, nminor, npatch = map(int, new_parts)
    if lmajor + 1 == nmajor and nminor == 0 and npatch == 0:
        return 'major'
    elif lmajor == nmajor and lminor + 1 == nminor and npatch == 0:
        return 'minor'
    elif lmajor == nmajor and lminor == nminor and lpatch + 1 == npatch:
        return 'patch'
    else:
        raise ValueError(
            'new_version is not a direct successor of last_version.')

def establish_new_version(version_location):
    '''Write the new release version to version_location.
    Check that this agrees with the bumped previous version.
    Return the new version.
    Args:
        - version_location: string, relative python module notation
        (e.g. PyHEADTAIL._version for PyHEADTAIL/_version.py)
    '''
    last_version = get_version(version_location)
    release_version = current_branch()[len(release_branch_prefix):]

    # make sure release_version incrementally succeeds last_version
    which_part_increases(last_version, release_version)

    vpath = version_location.replace('.', '/') + '.py'
    with open(vpath, 'wt') as vfile:
        vfile.write("__version__ = '" + release_version + "'\n")
    assert subprocess.call(["git", "add", vpath]) == 0
    assert subprocess.call(
        ["git", "commit", "-m", "release-script: bumping version file."]) == 0
    print ('*** The new release version has been bumped: PyHEADTAIL v'
           + release_version)
    return release_version

def current_branch():
    '''Return current git branch name.'''
    # get the current branch name, strip trailing whitespaces using rstrip()
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]).rstrip().decode("utf-8")

def open_release_branch(version):
    '''Create release/vX.Y.Z branch with the given version string.
    Push the new release branch upstream. Print output from git.
    '''
    branch_name = release_branch_prefix + version
    output = subprocess.check_output(
        ["git", "checkout", "-b", branch_name]
    ).rstrip().decode("utf-8")
    print (output)
    output = subprocess.check_output(
        ["git", "push", "--set-upstream", "origin", branch_name]
    ).rstrip().decode("utf-8")
    print (output)

def ensure_tests(test_script_location):
    '''Run test script and return whether they were successful.'''
    test_script = importlib.import_module(test_script_location)
    return test_script.run()

def is_worktree_dirty():
    '''Return True if the current git work tree is dirty, i.e.\ the
    status whether or not there are uncommitted changes.
    '''
    # integer (0: clean, 1: dirty)
    is_dirty = subprocess.call(['git', 'diff', '--quiet', 'HEAD', '--'])
    return bool(is_dirty)

def git_status():
    '''Print git status output.'''
    output = subprocess.check_output(
        ['git', 'status', 'HEAD']
    )
    print (output)


# DEFINE TWO STEPS FOR RELEASE PROCESS:

def init_release(part):
    '''Initialise release process.'''
    if not current_branch() == 'develop':
        raise EnvironmentError(
            'Releases can only be initiated from the develop branch!')
    if is_worktree_dirty():
        git_status()
        raise EnvironmentError('Release process can only be initiated on '
                               'a clean git repository. You have uncommitted '
                               'changes in your files, please fix this first.')
    current_version = get_version(version_location)
    new_version = bumpversion(current_version, part)
    open_release_branch(new_version)
    print ('*** The release process has been successfully initiated.\n'
           'Opening the pull request into master from the just created '
           'release branch.\n\n'
           'You may have to provide your github.com credentials '
           'to the following hub call.\n\n'
           'A text editor will open in which title and body of the pull '
           'request for the new release can be entered in the same '
           'manner as git commit message.')
    subprocess.call(["hub", "pull-request"])
    print ('\n*** Please check that the PyHEADTAIL tests run successfully.')
    print ('\n*** Initiated the release process. When you are ready to publish '
           'the release, run this command again.')

def finalise_release():
    '''Finalise release process.'''
    if is_worktree_dirty():
        git_status()
        raise EnvironmentError('Release process can only be initiated on '
                               'a clean git repository. You have uncommitted '
                               'changes in your files, please fix this first.')
    tests_successful = ensure_tests(test_script_location)
    if not tests_successful:
        raise EnvironmentError('The PyHEADTAIL tests fail. Please fix '
                               'the tests first!')
    print ('*** The PyHEADTAIL tests have successfully terminated.')
    new_version = establish_new_version(version_location)

    # make sure to push any possible release branch commits
    assert subprocess.call(["git", "push", "origin"]) == 0
    # --> might instead be done via git fetch and suggesting to push
    #     only if there are commits missing upstream

    # merge into master
    assert subprocess.call(["git", "checkout", "master"]) == 0
    rbranch = release_branch_prefix + new_version
    assert subprocess.call(
        ["git", "merge", "--no-ff", rbranch,
         "-m", "release-script: Merge branch '" + rbranch + "'"]) == 0

    # tag version on master commit
    assert subprocess.call(["git", "tag", "-a", "v" + new_version, "-m",
                            "'PyHEADTAIL v" + new_version + "'"]) == 0

    # push master release upstream
    assert subprocess.call(
        ["git", "push", "origin", "master", "--follow-tags"]) == 0

    # merge new master release back into develop
    assert subprocess.call(["git", "checkout", "develop"]) == 0
    assert subprocess.call(["git", "merge", "master"]) == 0
    assert subprocess.call(["git", "push", "origin", "develop"]) == 0

    # TO DO: publish github release (with text from pull request open in editor)

    # delete release branch
    assert subprocess.call(["git", "branch", "-d", rbranch]) == 0
    assert subprocess.call(["git", "push", "origin", ":" + rbranch]) == 0

# ALGORITHM FOR RELEASE PROCESS:
if __name__ == '__main__':
    print ('*** Current working directory:\n' + os.getcwd() + '\n')

    # are we on a release branch already?
    if not (current_branch()[:len(release_branch_prefix)] ==
            release_branch_prefix):
        args = parser.parse_args()
        init_release(args.part)
    else:
        finalise_release()
