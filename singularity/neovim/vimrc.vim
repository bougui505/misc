set mouse=
set wrap
autocmd BufNewFile *.sh 0r /etc/xdg/nvim/sh_template_file.sh
autocmd BufNewFile *.py 0r /etc/xdg/nvim/py_template_file.py
autocmd BufNewFile *.sh %s/<DATE>/\=strftime("%c")/
autocmd BufNewFile *.py %s/<DATE>/\=strftime("%c")/

" Make file executable if #! /bin/
function ModeChange()
  if getline(1) =~ "^#!"
    if getline(1) =~ "/bin/"
      silent !chmod a+x <afile>
    endif
  endif
endfunction
au BufWritePost * call ModeChange()
