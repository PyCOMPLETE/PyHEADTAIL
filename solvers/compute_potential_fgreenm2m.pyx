




<!DOCTYPE html>
<html class="   ">
  <head prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# object: http://ogp.me/ns/object# article: http://ogp.me/ns/article# profile: http://ogp.me/ns/profile#">
    <meta charset='utf-8'>
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    
    
    <title>PyHEADTAIL/solvers/compute_potential_fgreenm2m.pyx at kli · like2000/PyHEADTAIL</title>
    <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub" />
    <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub" />
    <link rel="apple-touch-icon" sizes="57x57" href="/apple-touch-icon-114.png" />
    <link rel="apple-touch-icon" sizes="114x114" href="/apple-touch-icon-114.png" />
    <link rel="apple-touch-icon" sizes="72x72" href="/apple-touch-icon-144.png" />
    <link rel="apple-touch-icon" sizes="144x144" href="/apple-touch-icon-144.png" />
    <meta property="fb:app_id" content="1401488693436528"/>

      <meta content="@github" name="twitter:site" /><meta content="summary" name="twitter:card" /><meta content="like2000/PyHEADTAIL" name="twitter:title" /><meta content="PyHEADTAIL - CERN HeadTail simulation code for simulation of multi-particle beam dynamics with collective effects" name="twitter:description" /><meta content="https://avatars2.githubusercontent.com/u/6576407?s=400" name="twitter:image:src" />
<meta content="GitHub" property="og:site_name" /><meta content="object" property="og:type" /><meta content="https://avatars2.githubusercontent.com/u/6576407?s=400" property="og:image" /><meta content="like2000/PyHEADTAIL" property="og:title" /><meta content="https://github.com/like2000/PyHEADTAIL" property="og:url" /><meta content="PyHEADTAIL - CERN HeadTail simulation code for simulation of multi-particle beam dynamics with collective effects" property="og:description" />

    <link rel="assets" href="https://assets-cdn.github.com/">
    <link rel="conduit-xhr" href="https://ghconduit.com:25035/">
    <link rel="xhr-socket" href="/_sockets" />

    <meta name="msapplication-TileImage" content="/windows-tile.png" />
    <meta name="msapplication-TileColor" content="#ffffff" />
    <meta name="selected-link" value="repo_source" data-pjax-transient />
      <meta name="google-analytics" content="UA-3769691-2">

    <meta content="collector.githubapp.com" name="octolytics-host" /><meta content="collector-cdn.github.com" name="octolytics-script-host" /><meta content="github" name="octolytics-app-id" /><meta content="808DF1F4:7CF5:83C2E4C:5385FD70" name="octolytics-dimension-request_id" /><meta content="7486834" name="octolytics-actor-id" /><meta content="michuschenk" name="octolytics-actor-login" /><meta content="ada033085a575d3badd06136a7b80672102998356cb686506f6c84b0896d9a46" name="octolytics-actor-hash" />
    

    
    
    <link rel="icon" type="image/x-icon" href="https://assets-cdn.github.com/favicon.ico" />

    <meta content="authenticity_token" name="csrf-param" />
<meta content="qNgrwoVzxRsBhI1p45vfPoqUgLlWkmmOWwqbGifL7XB8dA6FANOMH1PEfEgAoeZ3Lt/Ve0C2QE3ZSUYp8qKALg==" name="csrf-token" />

    <link href="https://assets-cdn.github.com/assets/github-1121bb0260c396426f82723a30b276d949f537a3.css" media="all" rel="stylesheet" type="text/css" />
    <link href="https://assets-cdn.github.com/assets/github2-712fa8c0954e275d6626ee28d9660e79c77e47c6.css" media="all" rel="stylesheet" type="text/css" />
    


    <meta http-equiv="x-pjax-version" content="3f16474b6e3a1ed04126225242cc563e">

      
  <meta name="description" content="PyHEADTAIL - CERN HeadTail simulation code for simulation of multi-particle beam dynamics with collective effects" />

  <meta content="6576407" name="octolytics-dimension-user_id" /><meta content="like2000" name="octolytics-dimension-user_login" /><meta content="16484363" name="octolytics-dimension-repository_id" /><meta content="like2000/PyHEADTAIL" name="octolytics-dimension-repository_nwo" /><meta content="true" name="octolytics-dimension-repository_public" /><meta content="false" name="octolytics-dimension-repository_is_fork" /><meta content="16484363" name="octolytics-dimension-repository_network_root_id" /><meta content="like2000/PyHEADTAIL" name="octolytics-dimension-repository_network_root_nwo" />
  <link href="https://github.com/like2000/PyHEADTAIL/commits/kli.atom" rel="alternate" title="Recent Commits to PyHEADTAIL:kli" type="application/atom+xml" />

  </head>


  <body class="logged_in  env-production linux vis-public page-blob">
    <a href="#start-of-content" tabindex="1" class="accessibility-aid js-skip-to-content">Skip to content</a>
    <div class="wrapper">
      
      
      
      


      <div class="header header-logged-in true">
  <div class="container clearfix">

    <a class="header-logo-invertocat" href="https://github.com/">
  <span class="mega-octicon octicon-mark-github"></span>
</a>

    
    <a href="/notifications" aria-label="You have no unread notifications" class="notification-indicator tooltipped tooltipped-s" data-hotkey="g n">
        <span class="mail-status all-read"></span>
</a>

      <div class="command-bar js-command-bar  in-repository">
          <form accept-charset="UTF-8" action="/search" class="command-bar-form" id="top_search_form" method="get">

<div class="commandbar">
  <span class="message"></span>
  <input type="text" data-hotkey="s" name="q" id="js-command-bar-field" placeholder="Search or type a command" tabindex="1" autocapitalize="off"
    
    data-username="michuschenk"
      data-repo="like2000/PyHEADTAIL"
      data-branch="kli"
      data-sha="4c57ed28e920c3530773a04a37643121b43f1286"
  >
  <div class="display hidden"></div>
</div>

    <input type="hidden" name="nwo" value="like2000/PyHEADTAIL" />

    <div class="select-menu js-menu-container js-select-menu search-context-select-menu">
      <span class="minibutton select-menu-button js-menu-target" role="button" aria-haspopup="true">
        <span class="js-select-button">This repository</span>
      </span>

      <div class="select-menu-modal-holder js-menu-content js-navigation-container" aria-hidden="true">
        <div class="select-menu-modal">

          <div class="select-menu-item js-navigation-item js-this-repository-navigation-item selected">
            <span class="select-menu-item-icon octicon octicon-check"></span>
            <input type="radio" class="js-search-this-repository" name="search_target" value="repository" checked="checked" />
            <div class="select-menu-item-text js-select-button-text">This repository</div>
          </div> <!-- /.select-menu-item -->

          <div class="select-menu-item js-navigation-item js-all-repositories-navigation-item">
            <span class="select-menu-item-icon octicon octicon-check"></span>
            <input type="radio" name="search_target" value="global" />
            <div class="select-menu-item-text js-select-button-text">All repositories</div>
          </div> <!-- /.select-menu-item -->

        </div>
      </div>
    </div>

  <span class="help tooltipped tooltipped-s" aria-label="Show command bar help">
    <span class="octicon octicon-question"></span>
  </span>


  <input type="hidden" name="ref" value="cmdform">

</form>
        <ul class="top-nav">
          <li class="explore"><a href="/explore">Explore</a></li>
            <li><a href="https://gist.github.com">Gist</a></li>
            <li><a href="/blog">Blog</a></li>
          <li><a href="https://help.github.com">Help</a></li>
        </ul>
      </div>

    


  <ul id="user-links">
    <li>
      <a href="/michuschenk" class="name">
        <img alt="michuschenk" class=" js-avatar" data-user="7486834" height="20" src="https://avatars3.githubusercontent.com/u/7486834?s=140" width="20" /> michuschenk
      </a>
    </li>

    <li class="new-menu dropdown-toggle js-menu-container">
      <a href="#" class="js-menu-target tooltipped tooltipped-s" aria-label="Create new...">
        <span class="octicon octicon-plus"></span>
        <span class="dropdown-arrow"></span>
      </a>

      <div class="new-menu-content js-menu-content">
      </div>
    </li>

    <li>
      <a href="/settings/profile" id="account_settings"
        class="tooltipped tooltipped-s"
        aria-label="Account settings ">
        <span class="octicon octicon-tools"></span>
      </a>
    </li>
    <li>
      <form class="logout-form" action="/logout" method="post">
        <button class="sign-out-button tooltipped tooltipped-s" aria-label="Sign out">
          <span class="octicon octicon-sign-out"></span>
        </button>
      </form>
    </li>

  </ul>

<div class="js-new-dropdown-contents hidden">
  

<ul class="dropdown-menu">
  <li>
    <a href="/new"><span class="octicon octicon-repo"></span> New repository</a>
  </li>
  <li>
    <a href="/organizations/new"><span class="octicon octicon-organization"></span> New organization</a>
  </li>


    <li class="section-title">
      <span title="like2000/PyHEADTAIL">This repository</span>
    </li>
      <li>
        <a href="/like2000/PyHEADTAIL/issues/new"><span class="octicon octicon-issue-opened"></span> New issue</a>
      </li>
</ul>

</div>


    
  </div>
</div>

      

        



      <div id="start-of-content" class="accessibility-aid"></div>
          <div class="site" itemscope itemtype="http://schema.org/WebPage">
    <div id="js-flash-container">
      
    </div>
    <div class="pagehead repohead instapaper_ignore readability-menu">
      <div class="container">
        

<ul class="pagehead-actions">

    <li class="subscription">
      <form accept-charset="UTF-8" action="/notifications/subscribe" class="js-social-container" data-autosubmit="true" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="xQBXtZZK/HGlTQ3/jv0knQearYeiV+1k/xRSedpkpfllUYphyl2qCnOkkrOCvKuxbRwlLv0jdwf+tExHDCGKxg==" /></div>  <input id="repository_id" name="repository_id" type="hidden" value="16484363" />

    <div class="select-menu js-menu-container js-select-menu">
      <a class="social-count js-social-count" href="/like2000/PyHEADTAIL/watchers">
        8
      </a>
      <span class="minibutton select-menu-button with-count js-menu-target" role="button" tabindex="0" aria-haspopup="true">
        <span class="js-select-button">
          <span class="octicon octicon-eye"></span>
          Unwatch
        </span>
      </span>

      <div class="select-menu-modal-holder">
        <div class="select-menu-modal subscription-menu-modal js-menu-content" aria-hidden="true">
          <div class="select-menu-header">
            <span class="select-menu-title">Notification status</span>
            <span class="octicon octicon-x js-menu-close"></span>
          </div> <!-- /.select-menu-header -->

          <div class="select-menu-list js-navigation-container" role="menu">

            <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input id="do_included" name="do" type="radio" value="included" />
                <h4>Not watching</h4>
                <span class="description">You only receive notifications for conversations in which you participate or are @mentioned.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-eye"></span>
                  Watch
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

            <div class="select-menu-item js-navigation-item selected" role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input checked="checked" id="do_subscribed" name="do" type="radio" value="subscribed" />
                <h4>Watching</h4>
                <span class="description">You receive notifications for all conversations in this repository.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-eye"></span>
                  Unwatch
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

            <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <div class="select-menu-item-text">
                <input id="do_ignore" name="do" type="radio" value="ignore" />
                <h4>Ignoring</h4>
                <span class="description">You do not receive any notifications for conversations in this repository.</span>
                <span class="js-select-button-text hidden-select-button-text">
                  <span class="octicon octicon-mute"></span>
                  Stop ignoring
                </span>
              </div>
            </div> <!-- /.select-menu-item -->

          </div> <!-- /.select-menu-list -->

        </div> <!-- /.select-menu-modal -->
      </div> <!-- /.select-menu-modal-holder -->
    </div> <!-- /.select-menu -->

</form>
    </li>

  <li>
  

  <div class="js-toggler-container js-social-container starring-container ">

    <form accept-charset="UTF-8" action="/like2000/PyHEADTAIL/unstar" class="js-toggler-form starred" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="22pvUzjWqHc3QRV6TXnBkbjveEO+zIZMdeyUGDfQkarr6q1Vkzxm3j+Ygw/W7R1T/kk6qnuiWddF7lAj/dSZVA==" /></div>
      <button
        class="minibutton with-count js-toggler-target star-button"
        aria-label="Unstar this repository" title="Unstar like2000/PyHEADTAIL">
        <span class="octicon octicon-star"></span><span class="text">Unstar</span>
      </button>
        <a class="social-count js-social-count" href="/like2000/PyHEADTAIL/stargazers">
          0
        </a>
</form>
    <form accept-charset="UTF-8" action="/like2000/PyHEADTAIL/star" class="js-toggler-form unstarred" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="sLySxBOovrYw0WzNrk7dEaO/H0FNgH//6QuzfJbWWSXB7jpivrqCxnqttjmyJyhHUnp7nkCPHJ9nr+aI5mtQBA==" /></div>
      <button
        class="minibutton with-count js-toggler-target star-button"
        aria-label="Star this repository" title="Star like2000/PyHEADTAIL">
        <span class="octicon octicon-star"></span><span class="text">Star</span>
      </button>
        <a class="social-count js-social-count" href="/like2000/PyHEADTAIL/stargazers">
          0
        </a>
</form>  </div>

  </li>


        <li>
          <a href="/like2000/PyHEADTAIL/fork" class="minibutton with-count js-toggler-target fork-button lighter tooltipped-n" title="Fork your own copy of like2000/PyHEADTAIL to your account" aria-label="Fork your own copy of like2000/PyHEADTAIL to your account" rel="nofollow" data-method="post">
            <span class="octicon octicon-repo-forked"></span><span class="text">Fork</span>
          </a>
          <a href="/like2000/PyHEADTAIL/network" class="social-count">1</a>
        </li>


</ul>

        <h1 itemscope itemtype="http://data-vocabulary.org/Breadcrumb" class="entry-title public">
          <span class="repo-label"><span>public</span></span>
          <span class="mega-octicon octicon-repo"></span>
          <span class="author"><a href="/like2000" class="url fn" itemprop="url" rel="author"><span itemprop="title">like2000</span></a></span><!--
       --><span class="path-divider">/</span><!--
       --><strong><a href="/like2000/PyHEADTAIL" class="js-current-repository js-repo-home-link">PyHEADTAIL</a></strong>

          <span class="page-context-loader">
            <img alt="" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
          </span>

        </h1>
      </div><!-- /.container -->
    </div><!-- /.repohead -->

    <div class="container">
      <div class="repository-with-sidebar repo-container new-discussion-timeline js-new-discussion-timeline  ">
        <div class="repository-sidebar clearfix">
            

<div class="sunken-menu vertical-right repo-nav js-repo-nav js-repository-container-pjax js-octicon-loaders">
  <div class="sunken-menu-contents">
    <ul class="sunken-menu-group">
      <li class="tooltipped tooltipped-w" aria-label="Code">
        <a href="/like2000/PyHEADTAIL/tree/kli" aria-label="Code" class="selected js-selected-navigation-item sunken-menu-item" data-hotkey="g c" data-pjax="true" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches /like2000/PyHEADTAIL/tree/kli">
          <span class="octicon octicon-code"></span> <span class="full-word">Code</span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

        <li class="tooltipped tooltipped-w" aria-label="Issues">
          <a href="/like2000/PyHEADTAIL/issues" aria-label="Issues" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-hotkey="g i" data-selected-links="repo_issues /like2000/PyHEADTAIL/issues">
            <span class="octicon octicon-issue-opened"></span> <span class="full-word">Issues</span>
            <span class='counter'>0</span>
            <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>

      <li class="tooltipped tooltipped-w" aria-label="Pull Requests">
        <a href="/like2000/PyHEADTAIL/pulls" aria-label="Pull Requests" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-hotkey="g p" data-selected-links="repo_pulls /like2000/PyHEADTAIL/pulls">
            <span class="octicon octicon-git-pull-request"></span> <span class="full-word">Pull Requests</span>
            <span class='counter'>0</span>
            <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>


        <li class="tooltipped tooltipped-w" aria-label="Wiki">
          <a href="/like2000/PyHEADTAIL/wiki" aria-label="Wiki" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-hotkey="g w" data-selected-links="repo_wiki /like2000/PyHEADTAIL/wiki">
            <span class="octicon octicon-book"></span> <span class="full-word">Wiki</span>
            <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>        </li>
    </ul>
    <div class="sunken-menu-separator"></div>
    <ul class="sunken-menu-group">

      <li class="tooltipped tooltipped-w" aria-label="Pulse">
        <a href="/like2000/PyHEADTAIL/pulse" aria-label="Pulse" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="pulse /like2000/PyHEADTAIL/pulse">
          <span class="octicon octicon-pulse"></span> <span class="full-word">Pulse</span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

      <li class="tooltipped tooltipped-w" aria-label="Graphs">
        <a href="/like2000/PyHEADTAIL/graphs" aria-label="Graphs" class="js-selected-navigation-item sunken-menu-item" data-pjax="true" data-selected-links="repo_graphs repo_contributors /like2000/PyHEADTAIL/graphs">
          <span class="octicon octicon-graph"></span> <span class="full-word">Graphs</span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>

      <li class="tooltipped tooltipped-w" aria-label="Network">
        <a href="/like2000/PyHEADTAIL/network" aria-label="Network" class="js-selected-navigation-item sunken-menu-item js-disable-pjax" data-selected-links="repo_network /like2000/PyHEADTAIL/network">
          <span class="octicon octicon-repo-forked"></span> <span class="full-word">Network</span>
          <img alt="" class="mini-loader" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32.gif" width="16" />
</a>      </li>
    </ul>


  </div>
</div>

              <div class="only-with-full-nav">
                

  

<div class="clone-url open"
  data-protocol-type="http"
  data-url="/users/set_protocol?protocol_selector=http&amp;protocol_type=push">
  <h3><strong>HTTPS</strong> clone URL</h3>
  <div class="clone-url-box">
    <input type="text" class="clone js-url-field"
           value="https://github.com/like2000/PyHEADTAIL.git" readonly="readonly">
    <span class="url-box-clippy">
    <button aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="https://github.com/like2000/PyHEADTAIL.git" data-copied-hint="copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>

  

<div class="clone-url "
  data-protocol-type="ssh"
  data-url="/users/set_protocol?protocol_selector=ssh&amp;protocol_type=push">
  <h3><strong>SSH</strong> clone URL</h3>
  <div class="clone-url-box">
    <input type="text" class="clone js-url-field"
           value="git@github.com:like2000/PyHEADTAIL.git" readonly="readonly">
    <span class="url-box-clippy">
    <button aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="git@github.com:like2000/PyHEADTAIL.git" data-copied-hint="copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>

  

<div class="clone-url "
  data-protocol-type="subversion"
  data-url="/users/set_protocol?protocol_selector=subversion&amp;protocol_type=push">
  <h3><strong>Subversion</strong> checkout URL</h3>
  <div class="clone-url-box">
    <input type="text" class="clone js-url-field"
           value="https://github.com/like2000/PyHEADTAIL" readonly="readonly">
    <span class="url-box-clippy">
    <button aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="https://github.com/like2000/PyHEADTAIL" data-copied-hint="copied!" type="button"><span class="octicon octicon-clippy"></span></button>
    </span>
  </div>
</div>


<p class="clone-options">You can clone with
      <a href="#" class="js-clone-selector" data-protocol="http">HTTPS</a>,
      <a href="#" class="js-clone-selector" data-protocol="ssh">SSH</a>,
      or <a href="#" class="js-clone-selector" data-protocol="subversion">Subversion</a>.
  <span class="help tooltipped tooltipped-n" aria-label="Get help on which URL is right for you.">
    <a href="https://help.github.com/articles/which-remote-url-should-i-use">
    <span class="octicon octicon-question"></span>
    </a>
  </span>
</p>



                <a href="/like2000/PyHEADTAIL/archive/kli.zip"
                   class="minibutton sidebar-button"
                   aria-label="Download like2000/PyHEADTAIL as a zip file"
                   title="Download like2000/PyHEADTAIL as a zip file"
                   rel="nofollow">
                  <span class="octicon octicon-cloud-download"></span>
                  Download ZIP
                </a>
              </div>
        </div><!-- /.repository-sidebar -->

        <div id="js-repo-pjax-container" class="repository-content context-loader-container" data-pjax-container>
          


<a href="/like2000/PyHEADTAIL/blob/0de39559dde9756d062ab7505482da2e0a72803b/solvers/compute_potential_fgreenm2m.pyx" class="hidden js-permalink-shortcut" data-hotkey="y">Permalink</a>

<!-- blob contrib key: blob_contributors:v21:eb7db8342b22ad7a7fc2e3d991cf1018 -->

<p title="This is a placeholder element" class="js-history-link-replace hidden"></p>

<a href="/like2000/PyHEADTAIL/find/kli" data-pjax data-hotkey="t" class="js-show-file-finder" style="display:none">Show File Finder</a>

<div class="file-navigation">
  

<div class="select-menu js-menu-container js-select-menu" >
  <span class="minibutton select-menu-button js-menu-target" data-hotkey="w"
    data-master-branch="master"
    data-ref="kli"
    role="button" aria-label="Switch branches or tags" tabindex="0" aria-haspopup="true">
    <span class="octicon octicon-git-branch"></span>
    <i>branch:</i>
    <span class="js-select-button">kli</span>
  </span>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax aria-hidden="true">

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <span class="select-menu-title">Switch branches/tags</span>
        <span class="octicon octicon-x js-menu-close"></span>
      </div> <!-- /.select-menu-header -->

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Find or create a branch…" id="context-commitish-filter-field" class="js-filterable-field js-navigation-enable" placeholder="Find or create a branch…">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" class="js-select-menu-tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" class="js-select-menu-tab">Tags</a>
            </li>
          </ul>
        </div><!-- /.select-menu-tabs -->
      </div><!-- /.select-menu-filters -->

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/HBartosik/solvers/compute_potential_fgreenm2m.pyx"
                 data-name="HBartosik"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="HBartosik">HBartosik</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/PYlongitudinal/solvers/compute_potential_fgreenm2m.pyx"
                 data-name="PYlongitudinal"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="PYlongitudinal">PYlongitudinal</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/alasheen/solvers/compute_potential_fgreenm2m.pyx"
                 data-name="alasheen"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="alasheen">alasheen</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/dquartul/solvers/compute_potential_fgreenm2m.pyx"
                 data-name="dquartul"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="dquartul">dquartul</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/htimko/solvers/compute_potential_fgreenm2m.pyx"
                 data-name="htimko"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="htimko">htimko</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item selected">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/kli/solvers/compute_potential_fgreenm2m.pyx"
                 data-name="kli"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="kli">kli</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/master/solvers/compute_potential_fgreenm2m.pyx"
                 data-name="master"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="master">master</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/oeftiger/solvers/compute_potential_fgreenm2m.pyx"
                 data-name="oeftiger"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="oeftiger">oeftiger</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/rfq/solvers/compute_potential_fgreenm2m.pyx"
                 data-name="rfq"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="rfq">rfq</a>
            </div> <!-- /.select-menu-item -->
        </div>

          <form accept-charset="UTF-8" action="/like2000/PyHEADTAIL/branches" class="js-create-branch select-menu-item select-menu-new-item-form js-navigation-item js-new-item-form" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="HDF02xwfzLv+h2miLkMltALcQuy2kKoawnbw69aXS/27f8b9yJmZMSqOh+qxk9cNMuAeIUCKVbklkYdWf5VAPw==" /></div>
            <span class="octicon octicon-git-branch select-menu-item-icon"></span>
            <div class="select-menu-item-text">
              <h4>Create branch: <span class="js-new-item-name"></span></h4>
              <span class="description">from ‘kli’</span>
            </div>
            <input type="hidden" name="name" id="name" class="js-new-item-value">
            <input type="hidden" name="branch" id="branch" value="kli" />
            <input type="hidden" name="path" id="path" value="solvers/compute_potential_fgreenm2m.pyx" />
          </form> <!-- /.select-menu-item -->

      </div> <!-- /.select-menu-list -->

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div> <!-- /.select-menu-list -->

    </div> <!-- /.select-menu-modal -->
  </div> <!-- /.select-menu-modal-holder -->
</div> <!-- /.select-menu -->

  <div class="breadcrumb">
    <span class='repo-root js-repo-root'><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/like2000/PyHEADTAIL/tree/kli" data-branch="kli" data-direction="back" data-pjax="true" itemscope="url"><span itemprop="title">PyHEADTAIL</span></a></span></span><span class="separator"> / </span><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/like2000/PyHEADTAIL/tree/kli/solvers" data-branch="kli" data-direction="back" data-pjax="true" itemscope="url"><span itemprop="title">solvers</span></a></span><span class="separator"> / </span><strong class="final-path">compute_potential_fgreenm2m.pyx</strong> <button aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="solvers/compute_potential_fgreenm2m.pyx" data-copied-hint="copied!" type="button"><span class="octicon octicon-clippy"></span></button>
  </div>
</div>


  <div class="commit commit-loader file-history-tease js-deferred-content" data-url="/like2000/PyHEADTAIL/contributors/kli/solvers/compute_potential_fgreenm2m.pyx">
    Fetching contributors…

    <div class="participation">
      <p class="loader-loading"><img alt="" height="16" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32-EAF2F5.gif" width="16" /></p>
      <p class="loader-error">Cannot retrieve contributors at this time</p>
    </div>
  </div>

<div class="file-box">
  <div class="file">
    <div class="meta clearfix">
      <div class="info file-name">
        <span class="icon"><b class="octicon octicon-file-text"></b></span>
        <span class="mode" title="File Mode">file</span>
        <span class="meta-divider"></span>
          <span>52 lines (39 sloc)</span>
          <span class="meta-divider"></span>
        <span>1.748 kb</span>
      </div>
      <div class="actions">
        <div class="button-group">
                <a class="minibutton js-update-url-with-hash"
                   href="/like2000/PyHEADTAIL/edit/kli/solvers/compute_potential_fgreenm2m.pyx"
                   data-method="post" rel="nofollow" data-hotkey="e">Edit</a>
          <a href="/like2000/PyHEADTAIL/raw/kli/solvers/compute_potential_fgreenm2m.pyx" class="button minibutton " id="raw-url">Raw</a>
            <a href="/like2000/PyHEADTAIL/blame/kli/solvers/compute_potential_fgreenm2m.pyx" class="button minibutton js-update-url-with-hash">Blame</a>
          <a href="/like2000/PyHEADTAIL/commits/kli/solvers/compute_potential_fgreenm2m.pyx" class="button minibutton " rel="nofollow">History</a>
        </div><!-- /.button-group -->

            <a class="minibutton danger empty-icon"
               href="/like2000/PyHEADTAIL/delete/kli/solvers/compute_potential_fgreenm2m.pyx"
               data-method="post" data-test-id="delete-blob-file" rel="nofollow">

          Delete
        </a>
      </div><!-- /.actions -->
    </div>
      
  <div class="blob-wrapper data type-cython js-blob-data">
       <table class="file-code file-diff tab-size-8">
         <tr class="file-code-line">
           <td class="blob-line-nums">
             <span id="L1" rel="#L1">1</span>
<span id="L2" rel="#L2">2</span>
<span id="L3" rel="#L3">3</span>
<span id="L4" rel="#L4">4</span>
<span id="L5" rel="#L5">5</span>
<span id="L6" rel="#L6">6</span>
<span id="L7" rel="#L7">7</span>
<span id="L8" rel="#L8">8</span>
<span id="L9" rel="#L9">9</span>
<span id="L10" rel="#L10">10</span>
<span id="L11" rel="#L11">11</span>
<span id="L12" rel="#L12">12</span>
<span id="L13" rel="#L13">13</span>
<span id="L14" rel="#L14">14</span>
<span id="L15" rel="#L15">15</span>
<span id="L16" rel="#L16">16</span>
<span id="L17" rel="#L17">17</span>
<span id="L18" rel="#L18">18</span>
<span id="L19" rel="#L19">19</span>
<span id="L20" rel="#L20">20</span>
<span id="L21" rel="#L21">21</span>
<span id="L22" rel="#L22">22</span>
<span id="L23" rel="#L23">23</span>
<span id="L24" rel="#L24">24</span>
<span id="L25" rel="#L25">25</span>
<span id="L26" rel="#L26">26</span>
<span id="L27" rel="#L27">27</span>
<span id="L28" rel="#L28">28</span>
<span id="L29" rel="#L29">29</span>
<span id="L30" rel="#L30">30</span>
<span id="L31" rel="#L31">31</span>
<span id="L32" rel="#L32">32</span>
<span id="L33" rel="#L33">33</span>
<span id="L34" rel="#L34">34</span>
<span id="L35" rel="#L35">35</span>
<span id="L36" rel="#L36">36</span>
<span id="L37" rel="#L37">37</span>
<span id="L38" rel="#L38">38</span>
<span id="L39" rel="#L39">39</span>
<span id="L40" rel="#L40">40</span>
<span id="L41" rel="#L41">41</span>
<span id="L42" rel="#L42">42</span>
<span id="L43" rel="#L43">43</span>
<span id="L44" rel="#L44">44</span>
<span id="L45" rel="#L45">45</span>
<span id="L46" rel="#L46">46</span>
<span id="L47" rel="#L47">47</span>
<span id="L48" rel="#L48">48</span>
<span id="L49" rel="#L49">49</span>
<span id="L50" rel="#L50">50</span>
<span id="L51" rel="#L51">51</span>

           </td>
           <td class="blob-line-code"><div class="code-body highlight"><pre><div class='line' id='LC1'><span class="k">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></div><div class='line' id='LC2'><span class="k">cimport</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></div><div class='line' id='LC3'><br/></div><div class='line' id='LC4'><span class="k">from</span> <span class="nn">libc.math</span> <span class="k">cimport</span> <span class="n">log</span></div><div class='line' id='LC5'><span class="k">from</span> <span class="nn">cython.parallel</span> <span class="k">import</span> <span class="n">parallel</span><span class="p">,</span> <span class="n">prange</span></div><div class='line' id='LC6'><br/></div><div class='line' id='LC7'><br/></div><div class='line' id='LC8'><span class="c"># from types import MethodType</span></div><div class='line' id='LC9'><span class="c"># Point.incx = MethodType(incx, None, Point)</span></div><div class='line' id='LC10'><br/></div><div class='line' id='LC11'><br/></div><div class='line' id='LC12'><span class="k">def</span> <span class="nf">compute_potential_fgreenm2m</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">double</span><span class="p">[:,:]</span> <span class="n">x</span><span class="p">,</span> <span class="n">double</span><span class="p">[:,:]</span> <span class="n">y</span><span class="p">,</span></div><div class='line' id='LC13'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">double</span><span class="p">[:,:]</span> <span class="n">phi</span><span class="p">,</span> <span class="n">double</span><span class="p">[:,:]</span> <span class="n">rho</span><span class="p">):</span></div><div class='line' id='LC14'><br/></div><div class='line' id='LC15'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">int</span> <span class="nf">k</span><span class="p">,</span> <span class="nf">l</span><span class="p">,</span> <span class="nf">m</span><span class="p">,</span> <span class="nf">n</span></div><div class='line' id='LC16'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">int</span> <span class="nf">nx</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">nx</span><span class="p">,</span> <span class="n">ny</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">ny</span></div><div class='line' id='LC17'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">xi</span><span class="p">,</span> <span class="nf">yi</span><span class="p">,</span> <span class="nf">xj</span><span class="p">,</span> <span class="nf">yj</span><span class="p">,</span> <span class="nf">r2</span></div><div class='line' id='LC18'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">da</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">dx</span> <span class="o">*</span> <span class="bp">self</span><span class="o">.</span><span class="n">dy</span></div><div class='line' id='LC19'><br/></div><div class='line' id='LC20'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">for</span> <span class="n">k</span> <span class="ow">in</span> <span class="n">prange</span><span class="p">(</span><span class="n">ny</span><span class="p">,</span> <span class="k">nogil</span><span class="o">=</span><span class="bp">True</span><span class="p">,</span> <span class="n">num_threads</span><span class="o">=</span><span class="mf">2</span><span class="p">):</span></div><div class='line' id='LC21'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">for</span> <span class="n">l</span> <span class="ow">in</span> <span class="nb">xrange</span><span class="p">(</span><span class="n">nx</span><span class="p">):</span></div><div class='line' id='LC22'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">xi</span><span class="p">,</span> <span class="n">yi</span> <span class="o">=</span> <span class="n">x</span><span class="p">[</span><span class="n">k</span><span class="p">,</span><span class="n">l</span><span class="p">],</span> <span class="n">y</span><span class="p">[</span><span class="n">k</span><span class="p">,</span><span class="n">l</span><span class="p">]</span></div><div class='line' id='LC23'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">for</span> <span class="n">m</span> <span class="ow">in</span> <span class="nb">xrange</span><span class="p">(</span><span class="n">ny</span><span class="p">):</span></div><div class='line' id='LC24'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">for</span> <span class="n">n</span> <span class="ow">in</span> <span class="nb">xrange</span><span class="p">(</span><span class="n">nx</span><span class="p">):</span></div><div class='line' id='LC25'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">xj</span><span class="p">,</span> <span class="n">yj</span> <span class="o">=</span> <span class="n">x</span><span class="p">[</span><span class="n">m</span><span class="p">,</span><span class="n">n</span><span class="p">],</span> <span class="n">y</span><span class="p">[</span><span class="n">m</span><span class="p">,</span><span class="n">n</span><span class="p">]</span></div><div class='line' id='LC26'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">r2</span> <span class="o">=</span> <span class="p">(</span><span class="n">xi</span> <span class="o">-</span> <span class="n">xj</span><span class="p">)</span> <span class="o">**</span> <span class="mf">2</span> <span class="o">+</span> <span class="p">(</span><span class="n">yi</span> <span class="o">-</span> <span class="n">yj</span><span class="p">)</span> <span class="o">**</span> <span class="mf">2</span> <span class="o">+</span> <span class="mf">1e-8</span></div><div class='line' id='LC27'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">phi</span><span class="p">[</span><span class="n">k</span><span class="p">,</span><span class="n">l</span><span class="p">]</span> <span class="o">+=</span> <span class="p">(</span><span class="o">-</span><span class="n">rho</span><span class="p">[</span><span class="n">m</span><span class="p">,</span><span class="n">n</span><span class="p">]</span> <span class="o">*</span> <span class="n">da</span> <span class="o">*</span> <span class="mf">1</span> <span class="o">/</span> <span class="mf">2.</span> <span class="o">*</span> <span class="n">log</span><span class="p">(</span><span class="n">r2</span><span class="p">))</span></div><div class='line' id='LC28'><br/></div><div class='line' id='LC29'><span class="k">def</span> <span class="nf">compute_potential_fgreenp2m</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">double</span><span class="p">[::</span><span class="mf">1</span><span class="p">]</span> <span class="n">xp</span><span class="p">,</span> <span class="n">double</span><span class="p">[::</span><span class="mf">1</span><span class="p">]</span> <span class="n">yp</span><span class="p">,</span></div><div class='line' id='LC30'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">double</span><span class="p">[:,:]</span> <span class="n">xg</span><span class="p">,</span> <span class="n">double</span><span class="p">[:,:]</span> <span class="n">yg</span><span class="p">,</span> <span class="n">double</span><span class="p">[:,:]</span> <span class="n">phi</span><span class="p">,</span> <span class="n">double</span><span class="p">[:,:]</span> <span class="n">rho</span><span class="p">):</span></div><div class='line' id='LC31'><br/></div><div class='line' id='LC32'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">int</span> <span class="nf">i</span><span class="p">,</span> <span class="nf">k</span><span class="p">,</span> <span class="nf">l</span></div><div class='line' id='LC33'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">int</span> <span class="nf">n</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">xp</span><span class="p">),</span> <span class="n">nx</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">nx</span><span class="p">,</span> <span class="n">ny</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">ny</span></div><div class='line' id='LC34'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">xg0</span><span class="p">,</span> <span class="nf">yg0</span><span class="p">,</span> <span class="nf">xp0</span><span class="p">,</span> <span class="nf">yp0</span><span class="p">,</span> <span class="nf">r2</span></div><div class='line' id='LC35'><br/></div><div class='line' id='LC36'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">for</span> <span class="n">k</span> <span class="ow">in</span> <span class="nb">xrange</span><span class="p">(</span><span class="n">ny</span><span class="p">):</span></div><div class='line' id='LC37'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">for</span> <span class="n">l</span> <span class="ow">in</span> <span class="nb">xrange</span><span class="p">(</span><span class="n">nx</span><span class="p">):</span></div><div class='line' id='LC38'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">xg0</span><span class="p">,</span> <span class="n">yg0</span> <span class="o">=</span> <span class="n">xg</span><span class="p">[</span><span class="n">k</span><span class="p">,</span><span class="n">l</span><span class="p">],</span> <span class="n">yg</span><span class="p">[</span><span class="n">k</span><span class="p">,</span><span class="n">l</span><span class="p">]</span></div><div class='line' id='LC39'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">xrange</span><span class="p">(</span><span class="n">n</span><span class="p">):</span></div><div class='line' id='LC40'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">xp0</span><span class="p">,</span> <span class="n">yp0</span> <span class="o">=</span> <span class="n">xp</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">yp</span><span class="p">[</span><span class="n">i</span><span class="p">]</span></div><div class='line' id='LC41'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">r2</span> <span class="o">=</span> <span class="p">(</span><span class="n">xg0</span> <span class="o">-</span> <span class="n">xp0</span><span class="p">)</span> <span class="o">**</span> <span class="mf">2</span> <span class="o">+</span> <span class="p">(</span><span class="n">yg0</span> <span class="o">-</span> <span class="n">xp0</span><span class="p">)</span> <span class="o">**</span> <span class="mf">2</span> <span class="o">+</span> <span class="mf">1e-8</span></div><div class='line' id='LC42'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">phi</span><span class="p">[</span><span class="n">k</span><span class="p">,</span><span class="n">l</span><span class="p">]</span> <span class="o">+=</span> <span class="o">-</span> <span class="mf">1</span> <span class="o">/</span> <span class="mf">2.</span> <span class="o">*</span> <span class="n">log</span><span class="p">(</span><span class="n">r2</span><span class="p">)</span></div><div class='line' id='LC43'><br/></div><div class='line' id='LC44'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># for i in xrange(n_macroparticles):</span></div><div class='line' id='LC45'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c">#     xi, yi = x[i], y[i]</span></div><div class='line' id='LC46'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c">#     for j in xrange(n_macroparticles):</span></div><div class='line' id='LC47'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c">#         xj, yj = x[j], y[j]</span></div><div class='line' id='LC48'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c">#         r2 = (xi - xj) ** 2 + (yi - yj) ** 2 + 1e-6</span></div><div class='line' id='LC49'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c">#         phi[i] += (-rho[j] * da * 1 / 2 * np.log(r2)) # * self.dx * self.dy / (2 * np.pi))</span></div><div class='line' id='LC50'><br/></div><div class='line' id='LC51'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># self.phi = np.reshape(phi, (ny, nx))</span></div></pre></div></td>
         </tr>
       </table>
  </div>

  </div>
</div>

<a href="#jump-to-line" rel="facebox[.linejump]" data-hotkey="l" class="js-jump-to-line" style="display:none">Jump to Line</a>
<div id="jump-to-line" style="display:none">
  <form accept-charset="UTF-8" class="js-jump-to-line-form">
    <input class="linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" autofocus>
    <button type="submit" class="button">Go</button>
  </form>
</div>

        </div>

      </div><!-- /.repo-container -->
      <div class="modal-backdrop"></div>
    </div><!-- /.container -->
  </div><!-- /.site -->


    </div><!-- /.wrapper -->

      <div class="container">
  <div class="site-footer">
    <ul class="site-footer-links right">
      <li><a href="https://status.github.com/">Status</a></li>
      <li><a href="http://developer.github.com">API</a></li>
      <li><a href="http://training.github.com">Training</a></li>
      <li><a href="http://shop.github.com">Shop</a></li>
      <li><a href="/blog">Blog</a></li>
      <li><a href="/about">About</a></li>

    </ul>

    <a href="/">
      <span class="mega-octicon octicon-mark-github" title="GitHub"></span>
    </a>

    <ul class="site-footer-links">
      <li>&copy; 2014 <span title="0.07454s from github-fe124-cp1-prd.iad.github.net">GitHub</span>, Inc.</li>
        <li><a href="/site/terms">Terms</a></li>
        <li><a href="/site/privacy">Privacy</a></li>
        <li><a href="/security">Security</a></li>
        <li><a href="/contact">Contact</a></li>
    </ul>
  </div><!-- /.site-footer -->
</div><!-- /.container -->


    <div class="fullscreen-overlay js-fullscreen-overlay" id="fullscreen_overlay">
  <div class="fullscreen-container js-fullscreen-container">
    <div class="textarea-wrap">
      <textarea name="fullscreen-contents" id="fullscreen-contents" class="fullscreen-contents js-fullscreen-contents" placeholder="" data-suggester="fullscreen_suggester"></textarea>
    </div>
  </div>
  <div class="fullscreen-sidebar">
    <a href="#" class="exit-fullscreen js-exit-fullscreen tooltipped tooltipped-w" aria-label="Exit Zen Mode">
      <span class="mega-octicon octicon-screen-normal"></span>
    </a>
    <a href="#" class="theme-switcher js-theme-switcher tooltipped tooltipped-w"
      aria-label="Switch themes">
      <span class="octicon octicon-color-mode"></span>
    </a>
  </div>
</div>



    <div id="ajax-error-message" class="flash flash-error">
      <span class="octicon octicon-alert"></span>
      <a href="#" class="octicon octicon-x close js-ajax-error-dismiss"></a>
      Something went wrong with that request. Please try again.
    </div>


      <script crossorigin="anonymous" src="https://assets-cdn.github.com/assets/frameworks-5bef6dacd990ce272ec009917ceea0b9d96f84b7.js" type="text/javascript"></script>
      <script async="async" crossorigin="anonymous" src="https://assets-cdn.github.com/assets/github-b34ff5b5950e79300fa8719b7d4a66b0d8723688.js" type="text/javascript"></script>
      
      
  </body>
</html>

