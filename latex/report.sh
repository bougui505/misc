#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Apr  3 14:19:00 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

# Path for progress reports
PRPATH=$HOME/Documents/progress_reports

bk () {
    if [[ -f $1 ]]; then
	    MODIFTIMESTAMP=$(date +%Y%m%d-%H%M%S -d "$(stat -c %y $1)") 
	    BKFILENAME=${1}~${MODIFTIMESTAMP}~
	    [ ! -f $BKFILENAME ] && cp -i -a -av $1 $BKFILENAME
	fi
}

if [[ ! -d $PRPATH ]]; then
    mkdir $PRPATH
fi

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
    -l, --list list all reports in $PRPATH
    -r, --report directory name of the given report $PRPATH
    -i, --img image to integrate to the slides
    -t, --title title of the slide
    -R, --recompile recompile the given report (-r)
        --fix fix the tex file by copying the images to the local figures directory
              and adjusting accordingly the path in the tex file
EOF
}

while [ "$#" -gt 0 ]; do
    case $1 in
        -l|--list) exa --icons -lh -snew -d --no-permissions --no-filesize --no-user $PRPATH/*/ ;;
        -r|--report) REPORT="$PRPATH/$2"; shift ;;
        -i|--img) IMG="$2"; shift ;;
        -t|--title) TITLE="$2"; shift ;;
        -R|--recompile) RECOMPILE=1 ;;
        --fix) FIX=1 ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

open_report ()
{
    PROCESS_NAME="evince $REPORT/build/slides.pdf"
    if pgrep -xf "$PROCESS_NAME" > /dev/null ; then
        echo "Process $PROCESS_NAME exists."
    else
        evince $REPORT/build/slides.pdf &
    fi
}

if [[ ! -d $REPORT ]] && [[ ! -z $REPORT ]]; then
    mkdir $REPORT
    cp -v $DIRSCRIPT/beamer_header.sty $REPORT/
    cat << EOF > $REPORT/slides.tex
\documentclass[aspectratio=169]{beamer}
\usepackage{beamer_header}
\AtBeginSection[]{
  \begin{frame}
  \vfill
  \centering
  \begin{beamercolorbox}[sep=8pt,center,shadow=true,rounded=true]{title}
    \usebeamerfont{title}\insertsectionhead\par%
  \end{beamercolorbox}
  \vfill
  \end{frame}
}
\title{$REPORT:t progress report}
\date{$(date +"%Y/%m/%d")}
\begin{document}
\maketitle
\end{document}
EOF
    cd $REPORT
    bk build/slides.pdf
    latexmk -pdf -outdir=build slides.tex
    cd -
    open_report
fi

if [[ ! -z $IMG ]]; then
    IMG=$(realpath $IMG)
    CKSUM=$(echo $IMG | md5sum | awk '{print $1}')
    IMGEXT=$IMG:e
    echo "CKSUM=${CKSUM}"
    CWD=$(pwd)
    cd $REPORT
    if [[ ! -d figures ]]; then
        mkdir figures
    fi
    LOCALIMG="figures/$CKSUM.$IMGEXT"
    cp -f -a -v $IMG $LOCALIMG
    if grep "$LOCALIMG" slides.tex > /dev/null; then
        echo "Image $IMG already presents in $REPORT"
    else
        sed -i '/\end{document}/d' slides.tex
        cat << EOF >> slides.tex
\begin{frame}{$TITLE}
\centering
\includegraphics[width=\linewidth,height=0.75\textheight,keepaspectratio]{$LOCALIMG}
\blfootnote{\scriptsize $(date +"%Y/%m/%d"):\url{$IMG}}
\end{frame}
\end{document}
EOF
        bk build/slides.pdf
        latexmk -pdf -outdir=build slides.tex
    fi
    cd -
fi

if [[ $FIX -eq 1 ]]; then
    cd $REPORT
    pwd
    if [[ ! -d figures ]]; then
        mkdir figures
    fi
    grep "^\\\includegraphics" slides.tex \
        | grep -v "{figures/" \
        | awk -F"{|}" '{print $2}' \
        > $MYTMP/files.list
    for IMG in $(cat $MYTMP/files.list); do
        echo $IMG
        CKSUM=$(echo $IMG | md5sum | awk '{print $1}')
        IMGEXT=$IMG:e
        LOCALIMG="figures/$CKSUM.$IMGEXT"
        cp -f -a -v $IMG $LOCALIMG
        sed -i "/includegraphics/s,$IMG,$LOCALIMG," slides.tex
    done
    RECOMPILE=1
fi

if [[ $RECOMPILE -eq 1 ]]; then
    cd $REPORT
    bk build/slides.pdf
    latexmk -pdf -outdir=build slides.tex
fi

if [[ ! -z $REPORT ]] && [[ -d $REPORT ]]; then
    open_report
fi
