#!/usr/bin/python
'''
Algorithm to release PyHEADTAIL versions. The structure follows the development
workflow, cf. the PyHEADTAIL wiki:
https://github.com/PyCOMPLETE/PyHEADTAIL/wiki/Our-Development-Workflow

Requires git, hub (the github tool, https://hub.github.com/) and importlib
(included in python >=2.7) to be installed.

Conforms with PEP440 (especially the versioning needs to follow this).

@copyright: CERN
@date: 26.01.2017
@author: Adrian Oeftiger
'''

import argparse
import importlib # available from PyPI for Python <2.7
import os, subprocess

# python2/3 compatibility for raw_input/input:
if hasattr(__builtins__, 'raw_input'):
    input = raw_input

# CONFIG
version_location = 'PyHEADTAIL._version' # in python relative module notation
# (e.g. PyHEADTAIL._version for PyHEADTAIL/_version.py)
test_script_location = 'prepush' # in python relative module notation
release_branch_prefix = 'release/v' # prepended name of release branch
github_user = 'PyCOMPLETE'
github_repo = 'PyHEADTAIL'


parser = argparse.ArgumentParser(
    description=(
        'Release a new version of PyHEADTAIL in 2 steps:\n'
        '1. prepare new release from develop branch '
        '(requires release type argument "part")\n'
        '2. publish new release (requires to be on release branch already)\n'
        'optionally: release PyHEADTAIL to PyPI '
        '(requires to be on master branch)'),
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

def do_tag_and_version_match(version):
    '''Return whether the current git tag and the given version match.
    NB: returns True if and only if the current commit is tagged with
    the version tag.
    '''
    current_commit = subprocess.check_output(
        ['git', 'describe', '--dirty']).rstrip().decode("utf-8")
    return current_commit == 'v{}'.format(version)

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

def validate_release_version(version_location):
    '''Validate the new release version new_version by comparing the
    release branch name to the bumped previous version last_version,
    which is read from version_location.
    Raise an error if new_version is not a valid direct successor to
    last_version. Return new_version.
    '''
    last_version = get_version(version_location)
    release_version = current_branch()[len(release_branch_prefix):]

    # make sure release_version incrementally succeeds last_version
    which_part_increases(last_version, release_version)

    return release_version


def establish_new_version(version_location):
    '''Write the new release version to version_location.
    Check that this agrees with the bumped previous version.
    Return the new version.
    Args:
        - version_location: string, relative python module notation
        (e.g. PyHEADTAIL._version for PyHEADTAIL/_version.py)
    '''
    release_version = validate_release_version(version_location)

    vpath = version_location.replace('.', '/') + '.py'
    with open(vpath, 'wt') as vfile:
        vfile.write("__version__ = '" + release_version + "'\n")
    assert subprocess.call(["git", "add", vpath]) == 0
    assert subprocess.call(
        ["git", "commit", "-m", "release-script: bumping version file."]) == 0
    print ('*** The new release version has been bumped: PyHEADTAIL v'
           + release_version)
    return release_version

def ensure_hub_is_installed():
    '''Check whether hub (from github) is installed.
    If not, throw an error with an installation note.
    '''
    try:
        assert subprocess.call(["hub", "--version"]) == 0
    except (OSError, AssertionError):
        raise OSError(
            'The github command-line tool is needed for '
            'opening the pull request for the release. '
            'Please install hub from https://hub.github.com/ !'
        )

def ensure_gothub_is_installed():
    '''Check whether gothub (to draft github releases) is installed.
    If not, throw an error with an installation note.
    '''
    try:
        assert subprocess.call(["gothub", "--version"]) == 0
    except (OSError, AssertionError):
        raise OSError(
            'The gothub command-line tool is needed for '
            'drafting releases on github. '
            'Please install gothub from '
            'https://github.com/itchio/gothub !'
        )

def check_or_setup_github_OAuth_token():
    '''Check if github OAuth security token is set as an environment
    variable. If not ask for it and give instruction how to get it.
    The token is needed for gothub.
    '''
    if os.environ.get('GITHUB_TOKEN', None):
        print ('\n*** github OAuth security token found in $GITHUB_TOKEN.\n')
        return
    print (
        '\n*** No github OAuth security token found in $GITHUB_TOKEN.'
        ' You need the token for gothub to draft the release on github!'
        ' (Get the security token via github\'s website, cf.'
        '\n*** https://help.github.com/articles/creating-a-personal-access-'
        'token-for-the-command-line/ )\n')
    token = input('--> Please enter your github OAuth security token:\n')
    os.environ['GITHUB_TOKEN'] = token

def ensure_gitpulls_is_installed():
    '''Check whether the gem git-pulls (to get github pull requests) is
    installed.
    If not, throw an error with an installation note.
    '''
    try:
        assert subprocess.call(["git", "pulls"]) == 0
    except (OSError, AssertionError):
        raise OSError(
            'The gothub command-line tool is needed for '
            'checking the pull request text from the release. '
            'Please install git-pulls '
            'from https://github.com/schacon/git-pulls !'
        )

def check_release_tools():
    '''Return whether git-pulls and gothub are installed (needed for
    drafting the github release from CLI). If not, ask whether user
    wants to continue and draft the release manually (if this is not
    the case, raise exception!).
    If no github OAuth security token is set, ask for it.
    '''
    try:
        ensure_gitpulls_is_installed()
        ensure_gothub_is_installed()
        check_or_setup_github_OAuth_token()
        return True
    except OSError:
        answer = ''
        accept = ['y', 'yes', 'n', 'no']
        while answer not in accept:
            answer = input(
                '!!! You do not have all required tools installed to '
                'automatically draft a release. Do you want to continue '
                'and manually draft the release on github afterwards?\n'
                '[y/N] ').lower()
            if not answer:
                answer = 'n'
        if answer == 'n' or answer == 'no':
            raise
        else:
            return False

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

def get_pullrequest_message(release_version):
    '''Fetch message from open pull request corresponding to this
    release_version.
    '''
    fetched = subprocess.check_output(['git', 'pulls', 'update'])
    pr_number = None
    for line in fetched.split('\n'):
        if "PyCOMPLETE:release/v{}".format(release_version) in line:
            pr_number = line.split()[0]
            break
    if pr_number is None:
        raise EnvironmentError(
            'Could not find open pull request for this release version. '
            'Did you properly initiate the release process '
            '(step 1 in ./release.py --help)?')
    text = subprocess.check_output(['git', 'pulls', 'show', pr_number])
    output = []
    for line in text.split('\n')[5:]:
        if line != '------------':
            output.append(line)
        else:
            break
    output[0] = output[0][11:] # remove "Title    : "
    return '\n'.join(output)


# DEFINE TWO STEPS FOR RELEASE PROCESS:

def init_release(part):
    '''Initialise release process.'''
    if not current_branch() == 'develop' and not (
            current_branch()[:7] == 'hotfix/' and
            part == 'patch'):
        raise EnvironmentError(
            'Releases can only be initiated from the develop branch! '
            '(Releasing a patch is allowed from a hotfix/ branch as well.)')
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
    print ('\n\n*** Do not forget to review the pull request on github.com!')

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

    # all tools installed to automatically draft release?
    draft_release = check_release_tools()
    if draft_release:
        new_version = validate_release_version(version_location)
        message = get_pullrequest_message(new_version)

    # bump version file
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

    # delete release branch
    assert subprocess.call(["git", "branch", "-d", rbranch]) == 0
    assert subprocess.call(["git", "push", "origin", ":" + rbranch]) == 0

    # publish github release (with message from pull request)
    if draft_release:
        release_failed = subprocess.call(
            ['gothub', 'release', '-u', github_user, '-r', github_repo,
             '-t', 'v' + new_version,
             '-n', 'PyHEADTAIL v{}'.format(new_version),
             '-d', '{}'.format(message),
             '-c', 'master'])
        if release_failed:
            print ('*** Drafting the release via gothub failed. '
                   'Did you provide the correct github OAuth security '
                   'token via $GITHUB_TOKEN? '
                   'You may draft this release manually from the '
                   'github website.')
    else:
        print ('*** Remember to manually draft this release from the '
               'github website.')

def release_pip():
    '''Release current version from master branch to PyPI.'''
    if is_worktree_dirty():
        git_status()
        raise EnvironmentError('Release process can only be initiated on '
                               'a clean git repository. You have uncommitted '
                               'changes in your files, please fix this first.')
    if current_branch() != "master":
        raise EnvironmentError(
            'PyPI releases can only be initiated from the master branch!')
    current_version = get_version(version_location)
    if not do_tag_and_version_match(current_version):
        raise EnvironmentError(
            'the current master branch commit needs to be the tagged version '
            'which matches the version stored in the version file!')

    assert subprocess.call(['python', 'setup.py', 'sdist']) == 0
    assert subprocess.call(
        ['twine', 'upload', '-r', 'pypi',
         'dist/PyHEADTAIL-{}.tar.gz'.format(current_version)]) == 0
    assert subprocess.call(['rm', '-r', 'dist', 'PyHEADTAIL.egg-info']) == 0


# ALGORITHM FOR RELEASE PROCESS:
if __name__ == '__main__':
    print ('*** Current working directory:\n' + os.getcwd() + '\n')
    ensure_hub_is_installed()

    # are we on a release branch already?
    if not (current_branch()[:len(release_branch_prefix)] ==
            release_branch_prefix):
        if current_branch() == 'master':
            release_pip()
        else:
            args = parser.parse_args()
            init_release(args.part)
    else:
        finalise_release()
