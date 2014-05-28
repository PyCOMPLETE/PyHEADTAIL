




<!DOCTYPE html>
<html class="   ">
  <head prefix="og: http://ogp.me/ns# fb: http://ogp.me/ns/fb# object: http://ogp.me/ns/object# article: http://ogp.me/ns/article# profile: http://ogp.me/ns/profile#">
    <meta charset='utf-8'>
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    
    
    <title>PyHEADTAIL/solvers/grid_functions.pyx at kli · like2000/PyHEADTAIL</title>
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

    <meta content="collector.githubapp.com" name="octolytics-host" /><meta content="collector-cdn.github.com" name="octolytics-script-host" /><meta content="github" name="octolytics-app-id" /><meta content="808DF1F4:7CF5:83C433D:5385FD7B" name="octolytics-dimension-request_id" /><meta content="7486834" name="octolytics-actor-id" /><meta content="michuschenk" name="octolytics-actor-login" /><meta content="ada033085a575d3badd06136a7b80672102998356cb686506f6c84b0896d9a46" name="octolytics-actor-hash" />
    

    
    
    <link rel="icon" type="image/x-icon" href="https://assets-cdn.github.com/favicon.ico" />

    <meta content="authenticity_token" name="csrf-param" />
<meta content="77IYmunqeGjw36dmNShkpINYmJcSIJL+Z0U1HaG0UFh1ey5T8/fFuIP6Zh10CP61EaUZuB0s3K//tE1fRJkBig==" name="csrf-token" />

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
      <form accept-charset="UTF-8" action="/notifications/subscribe" class="js-social-container" data-autosubmit="true" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="y+3qbsly1192WNYY5WcQIKvBFfzaGZquEp+o4Pit4t913dojF60fWzJAsXighqVgFd+WK2b/ZLl6CVxPohZLQg==" /></div>  <input id="repository_id" name="repository_id" type="hidden" value="16484363" />

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

    <form accept-charset="UTF-8" action="/like2000/PyHEADTAIL/unstar" class="js-toggler-form starred" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="rPGzbKmsNmBoexCkNEluelSIvDFilBD3LmZim+dGzt5UJrWMqV8dbZCKNsmw/lMhdapH2NOoz1u5mm4J+3k0Hw==" /></div>
      <button
        class="minibutton with-count js-toggler-target star-button"
        aria-label="Unstar this repository" title="Unstar like2000/PyHEADTAIL">
        <span class="octicon octicon-star"></span><span class="text">Unstar</span>
      </button>
        <a class="social-count js-social-count" href="/like2000/PyHEADTAIL/stargazers">
          0
        </a>
</form>
    <form accept-charset="UTF-8" action="/like2000/PyHEADTAIL/star" class="js-toggler-form unstarred" data-remote="true" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="WqRUFn+GsmqWvbPdmtJHQgCQvQH8bAJmxWQDjdpjTFZ+NOt8Xg0uL6/ku8QH2UD60yFsZdeDeJbskb7slJjUrQ==" /></div>
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
          


<a href="/like2000/PyHEADTAIL/blob/0de39559dde9756d062ab7505482da2e0a72803b/solvers/grid_functions.pyx" class="hidden js-permalink-shortcut" data-hotkey="y">Permalink</a>

<!-- blob contrib key: blob_contributors:v21:a26296fd928aa220e2c46f05944dc14d -->

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
              <a href="/like2000/PyHEADTAIL/blob/HBartosik/solvers/grid_functions.pyx"
                 data-name="HBartosik"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="HBartosik">HBartosik</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/PYlongitudinal/solvers/grid_functions.pyx"
                 data-name="PYlongitudinal"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="PYlongitudinal">PYlongitudinal</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/alasheen/solvers/grid_functions.pyx"
                 data-name="alasheen"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="alasheen">alasheen</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/dquartul/solvers/grid_functions.pyx"
                 data-name="dquartul"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="dquartul">dquartul</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/htimko/solvers/grid_functions.pyx"
                 data-name="htimko"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="htimko">htimko</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item selected">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/kli/solvers/grid_functions.pyx"
                 data-name="kli"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="kli">kli</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/master/solvers/grid_functions.pyx"
                 data-name="master"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="master">master</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/oeftiger/solvers/grid_functions.pyx"
                 data-name="oeftiger"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="oeftiger">oeftiger</a>
            </div> <!-- /.select-menu-item -->
            <div class="select-menu-item js-navigation-item ">
              <span class="select-menu-item-icon octicon octicon-check"></span>
              <a href="/like2000/PyHEADTAIL/blob/rfq/solvers/grid_functions.pyx"
                 data-name="rfq"
                 data-skip-pjax="true"
                 rel="nofollow"
                 class="js-navigation-open select-menu-item-text js-select-button-text css-truncate-target"
                 title="rfq">rfq</a>
            </div> <!-- /.select-menu-item -->
        </div>

          <form accept-charset="UTF-8" action="/like2000/PyHEADTAIL/branches" class="js-create-branch select-menu-item select-menu-new-item-form js-navigation-item js-new-item-form" method="post"><div style="margin:0;padding:0;display:inline"><input name="authenticity_token" type="hidden" value="T4H+0L4yBP1M1TQBniE9WqDiL+AdnGJHi2bDY0MwUMvEeznHCFsyKkYfvFUVbKJfkEcABg+OaRrHmQHXalgi+g==" /></div>
            <span class="octicon octicon-git-branch select-menu-item-icon"></span>
            <div class="select-menu-item-text">
              <h4>Create branch: <span class="js-new-item-name"></span></h4>
              <span class="description">from ‘kli’</span>
            </div>
            <input type="hidden" name="name" id="name" class="js-new-item-value">
            <input type="hidden" name="branch" id="branch" value="kli" />
            <input type="hidden" name="path" id="path" value="solvers/grid_functions.pyx" />
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
    <span class='repo-root js-repo-root'><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/like2000/PyHEADTAIL/tree/kli" data-branch="kli" data-direction="back" data-pjax="true" itemscope="url"><span itemprop="title">PyHEADTAIL</span></a></span></span><span class="separator"> / </span><span itemscope="" itemtype="http://data-vocabulary.org/Breadcrumb"><a href="/like2000/PyHEADTAIL/tree/kli/solvers" data-branch="kli" data-direction="back" data-pjax="true" itemscope="url"><span itemprop="title">solvers</span></a></span><span class="separator"> / </span><strong class="final-path">grid_functions.pyx</strong> <button aria-label="copy to clipboard" class="js-zeroclipboard minibutton zeroclipboard-button" data-clipboard-text="solvers/grid_functions.pyx" data-copied-hint="copied!" type="button"><span class="octicon octicon-clippy"></span></button>
  </div>
</div>


  <div class="commit file-history-tease">
      <img alt="" class="main-avatar" height="24" src="https://2.gravatar.com/avatar/a78e5553170a0225e5989c0508074e8d?d=https%3A%2F%2Fassets-cdn.github.com%2Fimages%2Fgravatars%2Fgravatar-user-420.png&amp;r=x&amp;s=140" width="24" />
      <span class="author"><span>Kevin Shing Bruce Li</span></span>
      <time datetime="2014-05-04T23:40:27+02:00" is="relative-time" title-format="%Y-%m-%d %H:%M:%S %z" title="2014-05-04 23:40:27 +0200">May 04, 2014</time>
      <div class="commit-title">
          <a href="/like2000/PyHEADTAIL/commit/d6c3a7f90664c77c01251bacfa91b86265d50d13" class="message" data-pjax="true" title="First version of spacecharge class.">First version of spacecharge class.</a>
      </div>

    <div class="participation">
      <p class="quickstat"><a href="#blob_contributors_box" rel="facebox"><strong>0</strong>  contributors</a></p>
      
    </div>
    <div id="blob_contributors_box" style="display:none">
      <h2 class="facebox-header">Users who have contributed to this file</h2>
      <ul class="facebox-user-list">
      </ul>
    </div>
  </div>

<div class="file-box">
  <div class="file">
    <div class="meta clearfix">
      <div class="info file-name">
        <span class="icon"><b class="octicon octicon-file-text"></b></span>
        <span class="mode" title="File Mode">file</span>
        <span class="meta-divider"></span>
          <span>211 lines (178 sloc)</span>
          <span class="meta-divider"></span>
        <span>6.043 kb</span>
      </div>
      <div class="actions">
        <div class="button-group">
                <a class="minibutton js-update-url-with-hash"
                   href="/like2000/PyHEADTAIL/edit/kli/solvers/grid_functions.pyx"
                   data-method="post" rel="nofollow" data-hotkey="e">Edit</a>
          <a href="/like2000/PyHEADTAIL/raw/kli/solvers/grid_functions.pyx" class="button minibutton " id="raw-url">Raw</a>
            <a href="/like2000/PyHEADTAIL/blame/kli/solvers/grid_functions.pyx" class="button minibutton js-update-url-with-hash">Blame</a>
          <a href="/like2000/PyHEADTAIL/commits/kli/solvers/grid_functions.pyx" class="button minibutton " rel="nofollow">History</a>
        </div><!-- /.button-group -->

            <a class="minibutton danger empty-icon"
               href="/like2000/PyHEADTAIL/delete/kli/solvers/grid_functions.pyx"
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
<span id="L52" rel="#L52">52</span>
<span id="L53" rel="#L53">53</span>
<span id="L54" rel="#L54">54</span>
<span id="L55" rel="#L55">55</span>
<span id="L56" rel="#L56">56</span>
<span id="L57" rel="#L57">57</span>
<span id="L58" rel="#L58">58</span>
<span id="L59" rel="#L59">59</span>
<span id="L60" rel="#L60">60</span>
<span id="L61" rel="#L61">61</span>
<span id="L62" rel="#L62">62</span>
<span id="L63" rel="#L63">63</span>
<span id="L64" rel="#L64">64</span>
<span id="L65" rel="#L65">65</span>
<span id="L66" rel="#L66">66</span>
<span id="L67" rel="#L67">67</span>
<span id="L68" rel="#L68">68</span>
<span id="L69" rel="#L69">69</span>
<span id="L70" rel="#L70">70</span>
<span id="L71" rel="#L71">71</span>
<span id="L72" rel="#L72">72</span>
<span id="L73" rel="#L73">73</span>
<span id="L74" rel="#L74">74</span>
<span id="L75" rel="#L75">75</span>
<span id="L76" rel="#L76">76</span>
<span id="L77" rel="#L77">77</span>
<span id="L78" rel="#L78">78</span>
<span id="L79" rel="#L79">79</span>
<span id="L80" rel="#L80">80</span>
<span id="L81" rel="#L81">81</span>
<span id="L82" rel="#L82">82</span>
<span id="L83" rel="#L83">83</span>
<span id="L84" rel="#L84">84</span>
<span id="L85" rel="#L85">85</span>
<span id="L86" rel="#L86">86</span>
<span id="L87" rel="#L87">87</span>
<span id="L88" rel="#L88">88</span>
<span id="L89" rel="#L89">89</span>
<span id="L90" rel="#L90">90</span>
<span id="L91" rel="#L91">91</span>
<span id="L92" rel="#L92">92</span>
<span id="L93" rel="#L93">93</span>
<span id="L94" rel="#L94">94</span>
<span id="L95" rel="#L95">95</span>
<span id="L96" rel="#L96">96</span>
<span id="L97" rel="#L97">97</span>
<span id="L98" rel="#L98">98</span>
<span id="L99" rel="#L99">99</span>
<span id="L100" rel="#L100">100</span>
<span id="L101" rel="#L101">101</span>
<span id="L102" rel="#L102">102</span>
<span id="L103" rel="#L103">103</span>
<span id="L104" rel="#L104">104</span>
<span id="L105" rel="#L105">105</span>
<span id="L106" rel="#L106">106</span>
<span id="L107" rel="#L107">107</span>
<span id="L108" rel="#L108">108</span>
<span id="L109" rel="#L109">109</span>
<span id="L110" rel="#L110">110</span>
<span id="L111" rel="#L111">111</span>
<span id="L112" rel="#L112">112</span>
<span id="L113" rel="#L113">113</span>
<span id="L114" rel="#L114">114</span>
<span id="L115" rel="#L115">115</span>
<span id="L116" rel="#L116">116</span>
<span id="L117" rel="#L117">117</span>
<span id="L118" rel="#L118">118</span>
<span id="L119" rel="#L119">119</span>
<span id="L120" rel="#L120">120</span>
<span id="L121" rel="#L121">121</span>
<span id="L122" rel="#L122">122</span>
<span id="L123" rel="#L123">123</span>
<span id="L124" rel="#L124">124</span>
<span id="L125" rel="#L125">125</span>
<span id="L126" rel="#L126">126</span>
<span id="L127" rel="#L127">127</span>
<span id="L128" rel="#L128">128</span>
<span id="L129" rel="#L129">129</span>
<span id="L130" rel="#L130">130</span>
<span id="L131" rel="#L131">131</span>
<span id="L132" rel="#L132">132</span>
<span id="L133" rel="#L133">133</span>
<span id="L134" rel="#L134">134</span>
<span id="L135" rel="#L135">135</span>
<span id="L136" rel="#L136">136</span>
<span id="L137" rel="#L137">137</span>
<span id="L138" rel="#L138">138</span>
<span id="L139" rel="#L139">139</span>
<span id="L140" rel="#L140">140</span>
<span id="L141" rel="#L141">141</span>
<span id="L142" rel="#L142">142</span>
<span id="L143" rel="#L143">143</span>
<span id="L144" rel="#L144">144</span>
<span id="L145" rel="#L145">145</span>
<span id="L146" rel="#L146">146</span>
<span id="L147" rel="#L147">147</span>
<span id="L148" rel="#L148">148</span>
<span id="L149" rel="#L149">149</span>
<span id="L150" rel="#L150">150</span>
<span id="L151" rel="#L151">151</span>
<span id="L152" rel="#L152">152</span>
<span id="L153" rel="#L153">153</span>
<span id="L154" rel="#L154">154</span>
<span id="L155" rel="#L155">155</span>
<span id="L156" rel="#L156">156</span>
<span id="L157" rel="#L157">157</span>
<span id="L158" rel="#L158">158</span>
<span id="L159" rel="#L159">159</span>
<span id="L160" rel="#L160">160</span>
<span id="L161" rel="#L161">161</span>
<span id="L162" rel="#L162">162</span>
<span id="L163" rel="#L163">163</span>
<span id="L164" rel="#L164">164</span>
<span id="L165" rel="#L165">165</span>
<span id="L166" rel="#L166">166</span>
<span id="L167" rel="#L167">167</span>
<span id="L168" rel="#L168">168</span>
<span id="L169" rel="#L169">169</span>
<span id="L170" rel="#L170">170</span>
<span id="L171" rel="#L171">171</span>
<span id="L172" rel="#L172">172</span>
<span id="L173" rel="#L173">173</span>
<span id="L174" rel="#L174">174</span>
<span id="L175" rel="#L175">175</span>
<span id="L176" rel="#L176">176</span>
<span id="L177" rel="#L177">177</span>
<span id="L178" rel="#L178">178</span>
<span id="L179" rel="#L179">179</span>
<span id="L180" rel="#L180">180</span>
<span id="L181" rel="#L181">181</span>
<span id="L182" rel="#L182">182</span>
<span id="L183" rel="#L183">183</span>
<span id="L184" rel="#L184">184</span>
<span id="L185" rel="#L185">185</span>
<span id="L186" rel="#L186">186</span>
<span id="L187" rel="#L187">187</span>
<span id="L188" rel="#L188">188</span>
<span id="L189" rel="#L189">189</span>
<span id="L190" rel="#L190">190</span>
<span id="L191" rel="#L191">191</span>
<span id="L192" rel="#L192">192</span>
<span id="L193" rel="#L193">193</span>
<span id="L194" rel="#L194">194</span>
<span id="L195" rel="#L195">195</span>
<span id="L196" rel="#L196">196</span>
<span id="L197" rel="#L197">197</span>
<span id="L198" rel="#L198">198</span>
<span id="L199" rel="#L199">199</span>
<span id="L200" rel="#L200">200</span>
<span id="L201" rel="#L201">201</span>
<span id="L202" rel="#L202">202</span>
<span id="L203" rel="#L203">203</span>
<span id="L204" rel="#L204">204</span>
<span id="L205" rel="#L205">205</span>
<span id="L206" rel="#L206">206</span>
<span id="L207" rel="#L207">207</span>
<span id="L208" rel="#L208">208</span>
<span id="L209" rel="#L209">209</span>
<span id="L210" rel="#L210">210</span>

           </td>
           <td class="blob-line-code"><div class="code-body highlight"><pre><div class='line' id='LC1'><br/></div><div class='line' id='LC2'><br/></div><div class='line' id='LC3'><span class="k">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></div><div class='line' id='LC4'><span class="k">cimport</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span></div><div class='line' id='LC5'><br/></div><div class='line' id='LC6'><br/></div><div class='line' id='LC7'><span class="k">from</span> <span class="nn">libc.math</span> <span class="k">cimport</span> <span class="n">floor</span></div><div class='line' id='LC8'><span class="k">from</span> <span class="nn">cython.parallel</span> <span class="k">import</span> <span class="n">parallel</span><span class="p">,</span> <span class="n">prange</span></div><div class='line' id='LC9'><br/></div><div class='line' id='LC10'><br/></div><div class='line' id='LC11'><span class="c"># Try using numpy arrays</span></div><div class='line' id='LC12'><span class="c"># cdef fastgather(self, np.ndarray[double, ndim=1] x,</span></div><div class='line' id='LC13'><span class="c">#                      np.ndarray[double, ndim=1] y,</span></div><div class='line' id='LC14'><span class="c">#                      np.ndarray[double, ndim=2] rho):</span></div><div class='line' id='LC15'><span class="k">def</span> <span class="nf">gather_from</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">double</span><span class="p">[:]</span> <span class="n">x</span><span class="p">,</span> <span class="n">double</span><span class="p">[:]</span> <span class="n">y</span><span class="p">,</span> <span class="n">double</span><span class="p">[:,:]</span> <span class="n">rho</span><span class="p">):</span></div><div class='line' id='LC16'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="sd">&#39;&#39;&#39;</span></div><div class='line' id='LC17'><span class="sd">    Cell</span></div><div class='line' id='LC18'><span class="sd">    3 ------------ 4</span></div><div class='line' id='LC19'><span class="sd">    |     |        |</span></div><div class='line' id='LC20'><span class="sd">    |     |        |</span></div><div class='line' id='LC21'><span class="sd">    |     |        |</span></div><div class='line' id='LC22'><span class="sd">    |     |        |</span></div><div class='line' id='LC23'><span class="sd">    |-----x--------|</span></div><div class='line' id='LC24'><span class="sd">    |     |        |</span></div><div class='line' id='LC25'><span class="sd">    1 ------------ 2</span></div><div class='line' id='LC26'><span class="sd">    &#39;&#39;&#39;</span></div><div class='line' id='LC27'><br/></div><div class='line' id='LC28'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># Line charge density</span></div><div class='line' id='LC29'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># lambda_e = self.density / self.n_macroparticles * (max_x - min_x) * (max_y - min_y)</span></div><div class='line' id='LC30'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># lambda_p = bunch.n_particles / bunch.n_macroparticles / dz;</span></div><div class='line' id='LC31'><br/></div><div class='line' id='LC32'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># Initialise</span></div><div class='line' id='LC33'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">rho</span><span class="p">[:]</span> <span class="o">=</span> <span class="mf">0</span></div><div class='line' id='LC34'><br/></div><div class='line' id='LC35'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># On regular mesh</span></div><div class='line' id='LC36'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">x0</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">x</span><span class="p">[</span><span class="mf">0</span><span class="p">,</span><span class="mf">0</span><span class="p">]</span></div><div class='line' id='LC37'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">x1</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">x</span><span class="p">[</span><span class="mf">0</span><span class="p">,</span><span class="mf">1</span><span class="p">]</span></div><div class='line' id='LC38'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">dx</span> <span class="o">=</span> <span class="n">x1</span> <span class="o">-</span> <span class="n">x0</span></div><div class='line' id='LC39'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">y0</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">y</span><span class="p">[</span><span class="mf">0</span><span class="p">,</span><span class="mf">0</span><span class="p">]</span></div><div class='line' id='LC40'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">y1</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">y</span><span class="p">[</span><span class="mf">1</span><span class="p">,</span><span class="mf">0</span><span class="p">]</span></div><div class='line' id='LC41'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">dy</span> <span class="o">=</span> <span class="n">y1</span> <span class="o">-</span> <span class="n">y0</span></div><div class='line' id='LC42'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">dxi</span> <span class="o">=</span> <span class="mf">1</span> <span class="o">/</span> <span class="n">dx</span></div><div class='line' id='LC43'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">dyi</span> <span class="o">=</span> <span class="mf">1</span> <span class="o">/</span> <span class="n">dy</span></div><div class='line' id='LC44'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">ai</span> <span class="o">=</span> <span class="mf">1</span> <span class="o">/</span> <span class="p">(</span><span class="n">dx</span> <span class="o">*</span> <span class="n">dy</span><span class="p">)</span></div><div class='line' id='LC45'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># TODO: on adaptive mesh</span></div><div class='line' id='LC46'><br/></div><div class='line' id='LC47'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">int</span> <span class="nf">i</span><span class="p">,</span> <span class="nf">n</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">x</span><span class="p">)</span></div><div class='line' id='LC48'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">int</span> <span class="nf">ix</span><span class="p">,</span> <span class="nf">iy</span></div><div class='line' id='LC49'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">fx</span><span class="p">,</span> <span class="nf">fy</span></div><div class='line' id='LC50'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">a1</span><span class="p">,</span> <span class="nf">a2</span><span class="p">,</span> <span class="nf">a3</span><span class="p">,</span> <span class="nf">a4</span></div><div class='line' id='LC51'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">q</span> <span class="o">=</span> <span class="mf">1</span> <span class="c"># if taken into account as coefficient for kx in push</span></div><div class='line' id='LC52'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># for i in prange(n, nogil=True, num_threads=2):</span></div><div class='line' id='LC53'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">xrange</span><span class="p">(</span><span class="n">n</span><span class="p">):</span></div><div class='line' id='LC54'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">fx</span><span class="p">,</span> <span class="n">fy</span> <span class="o">=</span> <span class="p">(</span><span class="n">x</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">-</span> <span class="n">x0</span><span class="p">)</span> <span class="o">*</span> <span class="n">dxi</span><span class="p">,</span> <span class="p">(</span><span class="n">y</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">-</span> <span class="n">y0</span><span class="p">)</span> <span class="o">*</span> <span class="n">dyi</span></div><div class='line' id='LC55'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">ix</span><span class="p">,</span> <span class="n">iy</span> <span class="o">=</span> <span class="p">&lt;</span><span class="kt">int</span><span class="p">&gt;</span> <span class="n">fx</span><span class="p">,</span> <span class="p">&lt;</span><span class="kt">int</span><span class="p">&gt;</span> <span class="n">fy</span></div><div class='line' id='LC56'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">fx</span><span class="p">,</span> <span class="n">fy</span> <span class="o">=</span> <span class="n">fx</span> <span class="o">-</span> <span class="n">ix</span><span class="p">,</span> <span class="n">fy</span> <span class="o">-</span> <span class="n">iy</span></div><div class='line' id='LC57'><br/></div><div class='line' id='LC58'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">a1</span> <span class="o">=</span> <span class="p">(</span><span class="mf">1</span> <span class="o">-</span> <span class="n">fx</span><span class="p">)</span> <span class="o">*</span> <span class="p">(</span><span class="mf">1</span> <span class="o">-</span> <span class="n">fy</span><span class="p">)</span></div><div class='line' id='LC59'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">a2</span> <span class="o">=</span> <span class="n">fx</span> <span class="o">*</span> <span class="p">(</span><span class="mf">1</span> <span class="o">-</span> <span class="n">fy</span><span class="p">)</span></div><div class='line' id='LC60'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">a3</span> <span class="o">=</span> <span class="p">(</span><span class="mf">1</span> <span class="o">-</span> <span class="n">fx</span><span class="p">)</span> <span class="o">*</span> <span class="n">fy</span></div><div class='line' id='LC61'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">a4</span> <span class="o">=</span> <span class="n">fx</span> <span class="o">*</span> <span class="n">fy</span></div><div class='line' id='LC62'><br/></div><div class='line' id='LC63'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">rho</span><span class="p">[</span><span class="n">iy</span><span class="p">,</span> <span class="n">ix</span><span class="p">]</span> <span class="o">+=</span> <span class="n">a1</span> <span class="o">*</span> <span class="n">ai</span> <span class="o">*</span> <span class="n">q</span></div><div class='line' id='LC64'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">rho</span><span class="p">[</span><span class="n">iy</span> <span class="o">+</span> <span class="mf">1</span><span class="p">,</span> <span class="n">ix</span><span class="p">]</span> <span class="o">+=</span> <span class="n">a2</span> <span class="o">*</span> <span class="n">ai</span> <span class="o">*</span> <span class="n">q</span></div><div class='line' id='LC65'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">rho</span><span class="p">[</span><span class="n">iy</span><span class="p">,</span> <span class="n">ix</span> <span class="o">+</span> <span class="mf">1</span><span class="p">]</span> <span class="o">+=</span> <span class="n">a3</span> <span class="o">*</span> <span class="n">ai</span> <span class="o">*</span> <span class="n">q</span></div><div class='line' id='LC66'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">rho</span><span class="p">[</span><span class="n">iy</span> <span class="o">+</span> <span class="mf">1</span><span class="p">,</span> <span class="n">ix</span> <span class="o">+</span> <span class="mf">1</span><span class="p">]</span> <span class="o">+=</span> <span class="n">a4</span> <span class="o">*</span> <span class="n">ai</span> <span class="o">*</span> <span class="n">q</span></div><div class='line' id='LC67'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># remember: 1 / dz missing here</span></div><div class='line' id='LC68'><br/></div><div class='line' id='LC69'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># H, xedges, yedges = np.histogram2d(ix, iy, bins=self.rho.shape)</span></div><div class='line' id='LC70'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># self.rho += H</span></div><div class='line' id='LC71'><br/></div><div class='line' id='LC72'><span class="c"># def weight(self):</span></div><div class='line' id='LC73'><br/></div><div class='line' id='LC74'><span class="c">#     pass</span></div><div class='line' id='LC75'><br/></div><div class='line' id='LC76'><span class="c"># def fastscatter(self):</span></div><div class='line' id='LC77'><span class="c">#     &#39;&#39;&#39;</span></div><div class='line' id='LC78'><span class="c">#     Cell</span></div><div class='line' id='LC79'><span class="c">#     3 ------------ 4</span></div><div class='line' id='LC80'><span class="c">#     |     |        |</span></div><div class='line' id='LC81'><span class="c">#     |     |        |</span></div><div class='line' id='LC82'><span class="c">#     |     |        |</span></div><div class='line' id='LC83'><span class="c">#     |     |        |</span></div><div class='line' id='LC84'><span class="c">#     |-----x--------|</span></div><div class='line' id='LC85'><span class="c">#     |     |        |</span></div><div class='line' id='LC86'><span class="c">#     1 ------------ 2</span></div><div class='line' id='LC87'><span class="c">#     &#39;&#39;&#39;</span></div><div class='line' id='LC88'><br/></div><div class='line' id='LC89'><span class="c">#     # Initialise</span></div><div class='line' id='LC90'><span class="c">#     cdef double[:,:] rho = self.rho</span></div><div class='line' id='LC91'><span class="c">#     rho[:] = 0</span></div><div class='line' id='LC92'><br/></div><div class='line' id='LC93'><span class="c">#     # On regular mesh</span></div><div class='line' id='LC94'><span class="c">#     cdef double x0 = self.x[0,0]</span></div><div class='line' id='LC95'><span class="c">#     cdef double x1 = self.x[0,1]</span></div><div class='line' id='LC96'><span class="c">#     cdef double dx = x1 - x0</span></div><div class='line' id='LC97'><span class="c">#     cdef double y0 = self.y[0,0]</span></div><div class='line' id='LC98'><span class="c">#     cdef double y1 = self.y[1,0]</span></div><div class='line' id='LC99'><span class="c">#     cdef double dy = y1 - y0</span></div><div class='line' id='LC100'><span class="c">#     cdef double dxi = 1 / dx</span></div><div class='line' id='LC101'><span class="c">#     cdef double dyi = 1 / dy</span></div><div class='line' id='LC102'><span class="c">#     cdef double ai = 1 / (dx * dy)</span></div><div class='line' id='LC103'><span class="c">#     # TODO: on adaptive mesh</span></div><div class='line' id='LC104'><br/></div><div class='line' id='LC105'><span class="c">#     cdef double fx, fy</span></div><div class='line' id='LC106'><span class="c">#     cdef double a1, a2, a3, a4</span></div><div class='line' id='LC107'><span class="c">#     cdef int ix, iy</span></div><div class='line' id='LC108'><span class="c">#     cdef int i, n = len(x)</span></div><div class='line' id='LC109'><span class="c">#     cdef int lambda_ = 1 # number_of_particles / dz</span></div><div class='line' id='LC110'><br/></div><div class='line' id='LC111'><span class="c">#     x, y = bunch.x, bunch.y</span></div><div class='line' id='LC112'><span class="c">#     n = bunch.n_macroparticles</span></div><div class='line' id='LC113'><span class="c">#     lambda_p = bunch.n_particles / bunch.n_macroparticles / dz;</span></div><div class='line' id='LC114'><span class="c">#     # for i in prange(n, nogil=True, num_threads=2):</span></div><div class='line' id='LC115'><span class="c">#     for i in xrange(n):</span></div><div class='line' id='LC116'><span class="c">#         fx, fy = (x[i] - x0) * dxi, (y[i] - y0) * dyi</span></div><div class='line' id='LC117'><span class="c">#         ix, iy = &lt;int&gt; fx, &lt;int&gt; fy</span></div><div class='line' id='LC118'><span class="c">#         fx, fy = fx - ix, fy - iy</span></div><div class='line' id='LC119'><br/></div><div class='line' id='LC120'><span class="c">#         a1 = (1 - fx) * (1 - fy)</span></div><div class='line' id='LC121'><span class="c">#         a2 = fx * (1 - fy)</span></div><div class='line' id='LC122'><span class="c">#         a3 = (1 - fx) * fy</span></div><div class='line' id='LC123'><span class="c">#         a4 = fx * fy</span></div><div class='line' id='LC124'><br/></div><div class='line' id='LC125'><span class="c">#         # Scatter fields</span></div><div class='line' id='LC126'><span class="c">#         bunch.kx[i] = 1</span></div><div class='line' id='LC127'><span class="c">#         bunch.ky[i] = 1</span></div><div class='line' id='LC128'><span class="c"># #       t.kx[ip[j]] = (u.ex_g[k1] * a1 + u.ex_g[k2] * a2</span></div><div class='line' id='LC129'><span class="c"># #                      + u.ex_g[k3] * a3 + u.ex_g[k4] * a4);</span></div><div class='line' id='LC130'><span class="c"># #       t.ky[ip[j]] = (u.ey_g[k1] * a1 + u.ey_g[k2] * a2</span></div><div class='line' id='LC131'><span class="c"># #                      + u.ey_g[k3] * a3 + u.ey_g[k4] * a4);</span></div><div class='line' id='LC132'><span class="c"># #     }</span></div><div class='line' id='LC133'><br/></div><div class='line' id='LC134'><span class="c">#         rho[iy, ix] += a1 * ai * lambda_</span></div><div class='line' id='LC135'><span class="c">#         rho[iy + 1, ix] += a2 * ai * lambda_</span></div><div class='line' id='LC136'><span class="c">#         rho[iy, ix + 1] += a3 * ai * lambda_</span></div><div class='line' id='LC137'><span class="c">#         rho[iy + 1, ix + 1] += a4 * ai * lambda_</span></div><div class='line' id='LC138'><br/></div><div class='line' id='LC139'><span class="c">#     x, y = self.x, self.y</span></div><div class='line' id='LC140'><span class="c">#     n = self.n_macroparticles</span></div><div class='line' id='LC141'><span class="c">#     lambda_e = self.density / self.n_macroparticles * (max_x - min_x) * (max_y - min_y)</span></div><div class='line' id='LC142'><span class="c">#     # for i in prange(n, nogil=True, num_threads=2):</span></div><div class='line' id='LC143'><span class="c">#     for i in xrange(n):</span></div><div class='line' id='LC144'><span class="c">#         fx, fy = (x[i] - x0) * dxi, (y[i] - y0) * dyi</span></div><div class='line' id='LC145'><span class="c">#         ix, iy = &lt;int&gt; fx, &lt;int&gt; fy</span></div><div class='line' id='LC146'><span class="c">#         fx, fy = fx - ix, fy - iy</span></div><div class='line' id='LC147'><br/></div><div class='line' id='LC148'><span class="c">#         a1 = (1 - fx) * (1 - fy)</span></div><div class='line' id='LC149'><span class="c">#         a2 = fx * (1 - fy)</span></div><div class='line' id='LC150'><span class="c">#         a3 = (1 - fx) * fy</span></div><div class='line' id='LC151'><span class="c">#         a4 = fx * fy</span></div><div class='line' id='LC152'><br/></div><div class='line' id='LC153'><span class="c">#         # Scatter fields</span></div><div class='line' id='LC154'><span class="c">#         self.kx[i] = 1</span></div><div class='line' id='LC155'><span class="c">#         self.ky[i] = 1</span></div><div class='line' id='LC156'><span class="c"># #       u.kx[ip[j]] = (t.ex_g[k1] * a1 + t.ex_g[k2] * a2</span></div><div class='line' id='LC157'><span class="c"># #                      + t.ex_g[k3] * a3 + t.ex_g[k4] * a4);</span></div><div class='line' id='LC158'><span class="c"># #       u.ky[ip[j]] = (t.ey_g[k1] * a1 + t.ey_g[k2] * a2</span></div><div class='line' id='LC159'><span class="c"># #                      + t.ey_g[k3] * a3 + t.ey_g[k4] * a4);</span></div><div class='line' id='LC160'><br/></div><div class='line' id='LC161'><span class="k">def</span> <span class="nf">scatter_to</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">double</span><span class="p">[:,:]</span> <span class="n">ex</span><span class="p">,</span> <span class="n">double</span><span class="p">[:,:]</span> <span class="n">ey</span><span class="p">,</span> <span class="n">double</span><span class="p">[::</span><span class="mf">1</span><span class="p">]</span> <span class="n">x</span><span class="p">,</span> <span class="n">double</span><span class="p">[::</span><span class="mf">1</span><span class="p">]</span> <span class="n">y</span><span class="p">,</span> <span class="n">double</span><span class="p">[::</span><span class="mf">1</span><span class="p">]</span> <span class="n">kx</span><span class="p">,</span> <span class="n">double</span><span class="p">[::</span><span class="mf">1</span><span class="p">]</span> <span class="n">ky</span><span class="p">):</span></div><div class='line' id='LC162'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="sd">&#39;&#39;&#39;</span></div><div class='line' id='LC163'><span class="sd">    Cell</span></div><div class='line' id='LC164'><span class="sd">    3 ------------ 4</span></div><div class='line' id='LC165'><span class="sd">    |     |        |</span></div><div class='line' id='LC166'><span class="sd">    |     |        |</span></div><div class='line' id='LC167'><span class="sd">    |     |        |</span></div><div class='line' id='LC168'><span class="sd">    |     |        |</span></div><div class='line' id='LC169'><span class="sd">    |-----x--------|</span></div><div class='line' id='LC170'><span class="sd">    |     |        |</span></div><div class='line' id='LC171'><span class="sd">    1 ------------ 2</span></div><div class='line' id='LC172'><span class="sd">    &#39;&#39;&#39;</span></div><div class='line' id='LC173'><br/></div><div class='line' id='LC174'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># Initialise</span></div><div class='line' id='LC175'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># cdef double[:,:] ex = self.ex</span></div><div class='line' id='LC176'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># cdef double[:,:] ey = self.ey</span></div><div class='line' id='LC177'><br/></div><div class='line' id='LC178'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># On regular mesh</span></div><div class='line' id='LC179'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">x0</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">x</span><span class="p">[</span><span class="mf">0</span><span class="p">,</span><span class="mf">0</span><span class="p">]</span></div><div class='line' id='LC180'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">x1</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">x</span><span class="p">[</span><span class="mf">0</span><span class="p">,</span><span class="mf">1</span><span class="p">]</span></div><div class='line' id='LC181'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">dx</span> <span class="o">=</span> <span class="n">x1</span> <span class="o">-</span> <span class="n">x0</span></div><div class='line' id='LC182'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">y0</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">y</span><span class="p">[</span><span class="mf">0</span><span class="p">,</span><span class="mf">0</span><span class="p">]</span></div><div class='line' id='LC183'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">y1</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">y</span><span class="p">[</span><span class="mf">1</span><span class="p">,</span><span class="mf">0</span><span class="p">]</span></div><div class='line' id='LC184'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">dy</span> <span class="o">=</span> <span class="n">y1</span> <span class="o">-</span> <span class="n">y0</span></div><div class='line' id='LC185'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">dxi</span> <span class="o">=</span> <span class="mf">1</span> <span class="o">/</span> <span class="n">dx</span></div><div class='line' id='LC186'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">dyi</span> <span class="o">=</span> <span class="mf">1</span> <span class="o">/</span> <span class="n">dy</span></div><div class='line' id='LC187'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">ai</span> <span class="o">=</span> <span class="mf">1</span> <span class="o">/</span> <span class="p">(</span><span class="n">dx</span> <span class="o">*</span> <span class="n">dy</span><span class="p">)</span></div><div class='line' id='LC188'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># TODO: on adaptive mesh</span></div><div class='line' id='LC189'><br/></div><div class='line' id='LC190'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">int</span> <span class="nf">ix</span><span class="p">,</span> <span class="nf">iy</span></div><div class='line' id='LC191'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">fx</span><span class="p">,</span> <span class="nf">fy</span></div><div class='line' id='LC192'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">double</span> <span class="nf">a1</span><span class="p">,</span> <span class="nf">a2</span><span class="p">,</span> <span class="nf">a3</span><span class="p">,</span> <span class="nf">a4</span></div><div class='line' id='LC193'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">cdef</span> <span class="kt">int</span> <span class="nf">i</span><span class="p">,</span> <span class="nf">n</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">x</span><span class="p">)</span></div><div class='line' id='LC194'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># cdef double[::1] x = other.x</span></div><div class='line' id='LC195'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># cdef double[::1] y = other.y</span></div><div class='line' id='LC196'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># cdef double[::1] kx = other.kx</span></div><div class='line' id='LC197'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># cdef double[::1] ky = other.ky</span></div><div class='line' id='LC198'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="c"># for i in prange(n, nogil=True, num_threads=2):</span></div><div class='line' id='LC199'>&nbsp;&nbsp;&nbsp;&nbsp;<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">xrange</span><span class="p">(</span><span class="n">n</span><span class="p">):</span></div><div class='line' id='LC200'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">fx</span><span class="p">,</span> <span class="n">fy</span> <span class="o">=</span> <span class="p">(</span><span class="n">x</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">-</span> <span class="n">x0</span><span class="p">)</span> <span class="o">*</span> <span class="n">dxi</span><span class="p">,</span> <span class="p">(</span><span class="n">y</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">-</span> <span class="n">y0</span><span class="p">)</span> <span class="o">*</span> <span class="n">dyi</span></div><div class='line' id='LC201'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">ix</span><span class="p">,</span> <span class="n">iy</span> <span class="o">=</span> <span class="p">&lt;</span><span class="kt">int</span><span class="p">&gt;</span> <span class="n">fx</span><span class="p">,</span> <span class="p">&lt;</span><span class="kt">int</span><span class="p">&gt;</span> <span class="n">fy</span></div><div class='line' id='LC202'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">fx</span><span class="p">,</span> <span class="n">fy</span> <span class="o">=</span> <span class="n">fx</span> <span class="o">-</span> <span class="n">ix</span><span class="p">,</span> <span class="n">fy</span> <span class="o">-</span> <span class="n">iy</span></div><div class='line' id='LC203'><br/></div><div class='line' id='LC204'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">a1</span> <span class="o">=</span> <span class="p">(</span><span class="mf">1</span> <span class="o">-</span> <span class="n">fx</span><span class="p">)</span> <span class="o">*</span> <span class="p">(</span><span class="mf">1</span> <span class="o">-</span> <span class="n">fy</span><span class="p">)</span></div><div class='line' id='LC205'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">a2</span> <span class="o">=</span> <span class="n">fx</span> <span class="o">*</span> <span class="p">(</span><span class="mf">1</span> <span class="o">-</span> <span class="n">fy</span><span class="p">)</span></div><div class='line' id='LC206'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">a3</span> <span class="o">=</span> <span class="p">(</span><span class="mf">1</span> <span class="o">-</span> <span class="n">fx</span><span class="p">)</span> <span class="o">*</span> <span class="n">fy</span></div><div class='line' id='LC207'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">a4</span> <span class="o">=</span> <span class="n">fx</span> <span class="o">*</span> <span class="n">fy</span></div><div class='line' id='LC208'><br/></div><div class='line' id='LC209'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">kx</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">ex</span><span class="p">[</span><span class="n">iy</span><span class="p">,</span> <span class="n">ix</span><span class="p">]</span> <span class="o">*</span> <span class="n">a1</span>  <span class="o">+</span> <span class="n">ex</span><span class="p">[</span><span class="n">iy</span><span class="p">,</span> <span class="n">ix</span> <span class="o">+</span> <span class="mf">1</span><span class="p">]</span> <span class="o">*</span> <span class="n">a2</span> <span class="o">+</span> <span class="n">ex</span><span class="p">[</span><span class="n">iy</span> <span class="o">+</span> <span class="mf">1</span><span class="p">,</span> <span class="n">ix</span><span class="p">]</span> <span class="o">*</span> <span class="n">a3</span> <span class="o">+</span> <span class="n">ex</span><span class="p">[</span><span class="n">iy</span> <span class="o">+</span> <span class="mf">1</span><span class="p">,</span> <span class="n">ix</span> <span class="o">+</span> <span class="mf">1</span><span class="p">]</span> <span class="o">*</span> <span class="n">a4</span></div><div class='line' id='LC210'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="n">ky</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">=</span> <span class="n">ey</span><span class="p">[</span><span class="n">iy</span><span class="p">,</span> <span class="n">ix</span><span class="p">]</span> <span class="o">*</span> <span class="n">a1</span>  <span class="o">+</span> <span class="n">ey</span><span class="p">[</span><span class="n">iy</span><span class="p">,</span> <span class="n">ix</span> <span class="o">+</span> <span class="mf">1</span><span class="p">]</span> <span class="o">*</span> <span class="n">a2</span> <span class="o">+</span> <span class="n">ey</span><span class="p">[</span><span class="n">iy</span> <span class="o">+</span> <span class="mf">1</span><span class="p">,</span> <span class="n">ix</span><span class="p">]</span> <span class="o">*</span> <span class="n">a3</span> <span class="o">+</span> <span class="n">ey</span><span class="p">[</span><span class="n">iy</span> <span class="o">+</span> <span class="mf">1</span><span class="p">,</span> <span class="n">ix</span> <span class="o">+</span> <span class="mf">1</span><span class="p">]</span> <span class="o">*</span> <span class="n">a4</span></div></pre></div></td>
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
      <li>&copy; 2014 <span title="0.09374s from github-fe123-cp1-prd.iad.github.net">GitHub</span>, Inc.</li>
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

