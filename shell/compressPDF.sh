#!/usr/bin/env bash
infilename=$1
outfilename=$2
#-dPDFSETTINGS=/screen   (screen-view-only quality, 72 dpi images)
#-dPDFSETTINGS=/ebook    (low quality, 150 dpi images)
#-dPDFSETTINGS=/printer  (high quality, 300 dpi images)
#-dPDFSETTINGS=/prepress (high quality, color preserving, 300 dpi imgs)
#-dPDFSETTINGS=/default  (almost identical to /screen)
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dEmbedAllFonts=true -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$outfilename $infilename
