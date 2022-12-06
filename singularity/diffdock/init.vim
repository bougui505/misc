"set termguicolors

set rtp+=/usr/local/etc/nvim

colorscheme github

" 4 space tab
set expandtab
set shiftwidth=4

" Fold indented code (see: https://stackoverflow.com/a/360634/1679629)
set foldmethod=indent
nnoremap <space> za
vnoremap <space> zf
set foldlevel=99  " Open file totally unfolded


" Map alt-b to Buffer
" noremap <M-b> :Buffer<CR>
cabbr B Buffer

" Omni completion
filetype plugin on
set omnifunc=syntaxcomplete#Complete
set completeopt=menuone,preview

" vim incremental search stop at end of file
set nowrapscan

set mouse=
" set mouse=a to activate in all mode
" set mouse=n : mouse support only in normal mode

" Uncomment the following to have Vim jump to the last position when
" reopening a file
if has("autocmd")
  au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
endif

" Command for pudb3 debugger:
command Pudb !tmux splitw -v zsh -i -l -c 'pudb3 %'
nmap <C-d> :Pudb<CR>

" Command for custom ctags for python
command Pytags !tmux splitw -v zsh -i -l -c 'pytags'

" Plugins will be downloaded under the specified directory. 
" Use the :PlugInstall command to install new plugins once added below
call plug#begin('/opt/nvim-plugged')
" Declare the list of plugins.
Plug 'airblade/vim-gitgutter'
Plug 'vim-scripts/vim-auto-save'
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'dense-analysis/ale'
Plug 'thaerkh/vim-workspace'
Plug 'iamcco/markdown-preview.nvim'
Plug 'https://github.com/coderifous/textobj-word-column.vim'
Plug 'Shougo/echodoc.vim'
Plug 'vim-scripts/autoproto.vim'
Plug 'davidhalter/jedi-vim'
Plug 'mattn/vim-gist'
Plug 'mattn/webapi-vim'
Plug 'aserebryakov/vim-todo-lists'
Plug 'preservim/tagbar'
Plug 'mbbill/undotree'
Plug 'preservim/nerdtree'
Plug 'ap/vim-buftabline'
Plug 'christoomey/vim-tmux-navigator'
Plug 'thaerkh/vim-workspace'
Plug 'pixelneo/vim-python-docstring'
" Plug 'jayli/vim-easycomplete'
Plug 'itchyny/vim-cursorword'
Plug 'easymotion/vim-easymotion'
Plug 'pseewald/vim-anyfold'
" Plug 'Townk/vim-autoclose'
" Plug 'heavenshell/vim-pydocstring', { 'do': 'make install', 'for': 'python' }
" List ends here. Plugins become visible to Vim after this call.
call plug#end()

autocmd Filetype * AnyFoldActivate               " activate for all filetypes
let g:anyfold_fold_comments=1

map <Leader> <Plug>(easymotion-prefix)

" vim-workspace
" let g:workspace_autocreate = 1
" let g:workspace_create_new_tabs = 1
" let g:workspace_persist_undo_history = 1

" tabnine
" set rtp+=~/source/tabnine-vim

" undotree
nnoremap tu :UndotreeToggle<CR>

" nerdtree
nnoremap tr :NERDTreeToggle<CR>
let NERDTreeShowBookmarks=1

" vim-todo-lists
let g:VimTodoListsDatesEnabled = 1
let g:VimTodoListsDatesFormat = "%Y-%m-%d"

" echodoc config
set cmdheight=2
let g:echodoc#enable_at_startup = 1

" markdown-preview config
" set to 1, nvim will open the preview window after entering the markdown buffer
" default: 0
let g:mkdp_auto_start = 1
" set to 1, the nvim will auto close current preview window when change
" from markdown buffer to another buffer
" default: 1
let g:mkdp_auto_close = 1

" vim-workspace config
" rather create a new buffer in the existing tab instead of creating a new tab
let g:workspace_create_new_tabs = 0
let g:workspace_autosave = 0
" let g:workspace_session_disable_on_args = 1

" slime plugin config
let g:slime_target = "tmux"
nmap <c-c><c-c> <Plug>SlimeRegionSend

" ale linter config
let g:ale_linters = {'python3': 'flake8'}
let g:ale_linters = {'python': 'flake8'}
let g:ale_fixers = {'python': 'yapf'}
" Set this variable to 1 to fix files when you save them.
let g:ale_fix_on_save = 1
let g:ale_lint_on_text_changed = 1
let g:ale_sign_priority = 0
nnoremap f :ALEFix<CR>
"let g:ale_lint_on_enter = 1

"tagbar configuration
noremap tt :TagbarToggle<CR>
let g:tagbar_sort = 0  " Do not sort tags
noremap t :TagbarOpen j<CR>

" list all available tag matches and query you instead of jumping to the first
" (see: https://superuser.com/a/679243/340948)
nnoremap <C-]> g<C-]>

let g:airline_theme='papercolor'
let g:airline#extensions#tagbar#enabled = 1

"let g:auto_save = 1  " enable AutoSave on Vim startup
let g:auto_save_in_insert_mode = 0  " do not save while in insert mode

"Kite completion config
" let g:kite_auto_complete=1
" nmap <silent> <buffer> K <Plug>(kite-docs)
"To have the preview window automatically closed once a completion has been
"inserted: (To have the preview window automatically closed once a completion
"has been inserted:)
"autocmd CompleteDone * if !pumvisible() | pclose | endif

let g:jedi#popup_on_dot = 0
let g:jedi#use_splits_not_buffers = 'winwidth'
let g:jedi#show_call_signatures = "0"

" Auto commands for scriptin
autocmd BufNewFile *.sh 0r ~/config/sh_template_file.sh
autocmd BufNewFile *.py 0r ~/config/py_template_file.py
autocmd BufNewFile .vimp 0put =\"$VIMPROJECT = ['', ]\"|
autocmd BufNewFile *.awk 0put =\"#!/usr/bin/awk -f\"|
  \ 1put=\"# -*- coding: UTF8 -*-\<nl>\<nl>\"|
  \ 2put=\"\n\"|
  \ 3put=\"# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr\"|
  \ 4put=\"# https://research.pasteur.fr/en/member/guillaume-bouvier/\"|
  \ 5put=\"# !!DATE!!\"|

autocmd BufNewFile *.c 0put=\"#include <stdio.h>\"|
  \ 1put=\"#include <stdlib.h>\"|$
autocmd BufNewFile *.java 
  \ 0put=\"import java.util.Scanner;\"|
  \ 2put=\"public class \".expand('%:t:r').\"{\"|
  \ 3put=\"}\"|$
au FileType java inoremap <buffer> <C-t> System.out.println();<esc>hi

" Make file executable if #! /bin/
function ModeChange()
  if getline(1) =~ "^#!"
    if getline(1) =~ "/bin/"
      silent !chmod a+x <afile>
    endif
  endif
endfunction
au BufWritePost * call ModeChange()

" Search for visually selected text (see: https://vim.fandom.com/wiki/Search_for_visually_selected_text)
vnoremap // y/<C-R>"<CR>

let g:gitgutter_override_sign_column_highlight = 0
let g:gitgutter_preview_win_floating = 0
"let g:gitgutter_highlight_lines = 1
" No line highlight for deletions  
" highlight GitGutterDeleteLine ctermfg=0 ctermbg=0
" Use fontawesome icons as signs
" let g:gitgutter_sign_added = '+'
" let g:gitgutter_sign_modified = '>'
" let g:gitgutter_sign_removed = '-'
" let g:gitgutter_sign_removed_first_line = '^'
" let g:gitgutter_sign_modified_removed = '<'

let g:gitgutter_sign_priority = 0
let g:gitgutter_sign_allow_clobber = 0
" colors for signs:
highlight SignColumn ctermbg=white
highlight GitGutterAdd ctermbg=Green
highlight GitGutterDelete ctermbg=Red
highlight GitGutterChange ctermbg=Blue
" Key map for hunks
nmap ghs <Plug>(GitGutterStageHunk)
nmap ghu <Plug>(GitGutterUndoHunk)
nmap ghp <Plug>(GitGutterPreviewHunk)
" Update sign column every quarter second
set updatetime=250

"Colorcolumn ruler
set colorcolumn=120
highlight ColorColumn ctermbg=lightgrey guibg=lightgrey

" Reload the buffer if file changed elsewhere
" See: https://unix.stackexchange.com/a/383044/68794
au CursorHold,CursorHoldI * checktime

" Mapping buffer navigation
" Shift of 1 as in a vim session buffer 1 store the session file...
map b1 :b2<cr>
map b2 :b3<cr>
map b3 :b4<cr>
map b4 :b5<cr>
map b5 :b6<cr>
map b6 :b7<cr>
map b7 :b8<cr>

" Update tags file when a file is written:
" autocmd BufWritePost *.py silent exec "!ctags -R ."

let g:tmux_navigator_no_mappings = 1

nnoremap <silent> <S-Left> :TmuxNavigateLeft<cr>
nnoremap <silent> <S-Down> :TmuxNavigateDown<cr>
nnoremap <silent> <S-Up> :TmuxNavigateUp<cr>
nnoremap <silent> <S-Right> :TmuxNavigateRight<cr>
