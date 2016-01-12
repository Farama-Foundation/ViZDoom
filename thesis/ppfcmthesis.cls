%
% This is a LaTeX class for master and bachelor theses following the guidelines
% of Faculty of Computing at Poznan University of Technology.
%
% This class is UNOFFICIAL and NOT ENDORSED by the university.
%
% (c) 2006--2011 Dawid Weiss, Poznan University of Technology
% (c) 2012--     Przemyslaw Wesolek, Poznan University of Technology
%
% Check for an up-to-date version of this template at:
% https://github.com/politechnika/put-latex
%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{ppfcmthesis}[2012/07/04 v0.98 PP FCM thesis class]

% v.0.98 Changed faculty name.
% v.0.97 Added \authortitle{} for people who really need to have "inż." in front of their names.
%        Dropped "r." from the title page in Polish.
% v.0.96 Fix for setspace package: emulation of new memoir commands for older memoir packages (tetex).
% v.0.95 Fix for setspace package emulated by memoir, causing exceptions in MikTeX 2.5  
% v.0.94 Added "bachelor" option for bachelor theses. 
% v.0.93 Varioref is loaded prior to hyperref.

% Declare bachelor thesis switch.
\newif\if@ppbachelor\@ppbachelorfalse

%
% Declare default options and load base package (memoir).
%
\DeclareOption{bachelor}
   {\@ppbachelortrue}

\ExecuteOptions{a4paper,pl,11pt,oneside,final}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax
\LoadClass{memoir}

%
% Require additional packages
%
\RequirePackage{babel}
\newif\if@pppolish
\iflanguage{polish}{\@pppolishtrue}{\@pppolishfalse}

\if@pppolish
\selectlanguage{polish}
\RequirePackage{indentfirst}
\addto\captionspolish{\renewcommand{\bibname}{Literatura}}   % Force "Literatura" instead of "Bibliografia"
\else
\selectlanguage{english}
\fi

\RequirePackage{graphicx}
\RequirePackage{setspace}
\RequirePackage{booktabs}

\ifpdf
	\DeclareGraphicsExtensions{.pdf,.png,.jpg}
\else
	\DeclareGraphicsExtensions{.eps}
\fi
\RequirePackage{fancyvrb}

%
% Configure Hyperref
%

\RequirePackage{url}
\RequirePackage{nameref}
\RequirePackage{varioref}
  
\ifpdf
        \usepackage[unicode,linktocpage,bookmarks]{hyperref}
\else
        \usepackage[dvips,unicode,linktocpage,bookmarks]{hyperref}
\fi
\hypersetup{
    pdffitwindow=true,
    bookmarksnumbered=true,
    bookmarksopen=true,
    colorlinks=true,
    pdfpagelayout=SinglePage,
    pdfpagemode=UseOutlines,
    pdfstartview=Fit,
    linkcolor=black,
    citecolor=black,
    anchorcolor=black,
    filecolor=black,
    menucolor=black,
    pagecolor=black,
    urlcolor=black,
    pdftitle={},
    pdfauthor={},
    pdfkeywords={}
}
\RequirePackage{memhfixc}

%
% Emulate setspace package for older versions of memoir.
%
\ifx\undefined\SingleSpacing
\RequirePackage{setspace}
\newcommand{\SingleSpacing}{\singlespacing}
\fi
\ifx\undefined\OnehalfSpacing
\newcommand{\OnehalfSpacing}{\onehalfspacing}
\fi

%
% Customize title page
%
\aliaspagestyle{title}{empty}
\date{}
\newcommand{\ppsupervisor}[1]{\def\@ppsupervisor{#1}}
\newcommand{\ppyear}[1]{\def\@ppyear{#1}}

\newcommand{\authortitle}[1]{\def\@authortitle{#1}}

\setlength{\droptitle}{0cm}

\preauthor{\begin{center}\large\@authortitle}
\postauthor{\end{center}}

\renewcommand{\maketitlehooka}{%
\begin{center}
\if@pppolish
Politechnika Pozna\'nska\\Wydzia{\l} Informatyki\\Instytut Informatyki
\else
Poznan University of Technology\\Faculty of Computing\\Institute of Computing Science
\fi\end{center}}

\pretitle{\vspace{2cm}\begin{center}%
\if@pppolish%
\if@ppbachelor Praca dyplomowa in{\.z}ynierska\else Praca dyplomowa magisterska\fi
\else
\if@ppbachelor Bachelor's thesis\else Master's thesis\fi
\fi\par\vspace{1cm}\Large\bfseries\MakeUppercase}
\posttitle{\end{center}}

\newcommand{\album}[1]{\unskip ,~#1}
\renewcommand{\and}{\\}

\renewcommand{\maketitlehookb}{\vspace{1cm}}

\renewcommand{\maketitlehookc}{\vfill\begin{center}
\if@pppolish
Promotor\\\@ppsupervisor
\else
Supervisor\\\@ppsupervisor
\fi\end{center}
\vspace{2cm}\begin{center}Pozna\'n, \@ppyear\if@pppolish\fi\end{center}
}

%
% Define colophon 
%
\newcommand{\ppcolophon}{%
\renewcommand{\and}{\unskip{}, }%
\renewcommand{\album}[1]{\unskip}%
\cleardoublepage\thispagestyle{empty}\vspace*{\fill}%
\footnotesize\SingleSpacing\noindent%
\raisebox{-1.3cm}{\includegraphics[height=1.5cm]{figures/template/logo-pp}}%
\hspace{1.5em}%
\begin{minipage}[t]{11cm}%
\vglue 0pt\copyright{} \@ppyear\ \@author

\medskip\noindent%
\if@pppolish%
Instytut Informatyki, Wydzia{\l} Informatyki\\%
Politechnika Pozna\'nska%
\else%
Poznan University of Technology\\Faculty of Computing\\Institute of Computing Science
\fi

\medskip\noindent%
\if@pppolish%
Sk{\l}ad przy u{\.z}yciu systemu \LaTeX{}.
\else%
Typeset using \LaTeX\ in Computer Modern.
\fi

\renewcommand{\and}{\unskip{} $\backslash$and }%
\medskip\noindent Bib\TeX{}:\\[1ex]
\tiny\texttt{%
@mastersthesis\{ VIZAI,\\
\hspace*{2em}author = "\@author",\\
\hspace*{2em}title = "\{\@title\}",\\
\hspace*{2em}school = "Poznan University of Technology",\\
\hspace*{2em}address = "Pozna\{$\backslash$'n\}, Poland",\\
\hspace*{2em}year = "\@ppyear",\\
\}
}
\end{minipage}
}

%
% Customize running titles and page style.
%

\makepagestyle{ppfcmthesis}
\makeevenfoot{ppfcmthesis}{}{}{}
\makeatletter
\newcommand{\@ppfcmthesismarks}{%
  \let\@mkboth\markboth
  \def\chaptermark##1{\markboth{##1}{##1}}    % left mark & right marks
  \def\sectionmark##1{\markright{%
    \ifnum \c@secnumdepth>\z@
      \thesection. \ %
    \fi
    ##1}}
  \def\tocmark{\markboth{\contentsname}{}}
  \def\lofmark{\markboth{\listfigurename}{}}
  \def\lotmark{\markboth{\listtablename}{}}
  \def\bibmark{\markboth{\bibname}{}}
  \def\indexmark{\markboth{\indexname}{}}
}
\makepsmarks{ppfcmthesis}{\@ppfcmthesismarks}
\makeatother
\makeevenhead{ppfcmthesis}{\thepage}{}{\normalfont\small\itshape\leftmark}
\makeoddhead{ppfcmthesis}{\normalfont\small\itshape\rightmark}{}{\thepage}

%
% Customize Table of Contents
%
\setcounter{tocdepth}{3}
\maxtocdepth{subsubsection}
\maxsecnumdepth{subsection}

%
% Customize book style
%
% \tightlists
\pagestyle{ppfcmthesis}
\setstocksize{297mm}{210mm}
\settrimmedsize{297mm}{210mm}{*}
\settrims{0pt}{0cm}
\setlrmarginsandblock{3.5cm}{2.5cm}{*}
\setulmarginsandblock{2.5cm}{2.5cm}{*}
\setmarginnotes{6pt}{50pt}{\onelineskip}
\setheadfoot{\onelineskip}{2\onelineskip}
\setheaderspaces{*}{2\onelineskip}{*}
\OnehalfSpacing

\changecaptionwidth
\captionwidth{.9\linewidth}
\captionnamefont{\footnotesize\scshape}
\captiontitlefont{\footnotesize}

\newcommand{\fcmfcaption}[2][\@empty]{
\captionnamefont{\footnotesize\bfseries}%
\captiondelim{.~}
\captionstyle{}
\captiontitlefont{\footnotesize}
\ifx #1\@empty \caption{#2}\else \caption[#1]{#2}}

\newcommand{\fcmtcaption}[2][\@empty]{
\captionnamefont{\footnotesize\bfseries\hfill}%
\captiondelim{}
\captionstyle{\raggedleft\\}
\captiontitlefont{\footnotesize\normalfont}
\ifx #1\@empty \caption{#2}\else \caption[#1]{#2}}

\def\thesubsubsection {\thesubsection .\arabic{subsubsection}}

\checkandfixthelayout

%
% Macros 
%

% English
\newcommand{\termdef}[1]{\emph{#1}}
\newcommand{\acronym}[1]{\emph{\MakeUppercase{#1}}}

% Polish
\newcommand{\definicja}[1]{\termdef{#1}}
\newcommand{\akronim}[1]{\acronym{#1}}
\newcommand{\english}[1]{ang.~\emph{#1}}

% Table head
\newcolumntype{L}{c}\newcommand{\tabhead}[1]{\multicolumn{1}{L}{#1}}
