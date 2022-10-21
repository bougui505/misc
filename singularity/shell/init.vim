set rtp+=/usr/local/etc/nvim

set foldmethod=indent
nnoremap <space> za
vnoremap <space> zf
set foldlevel=99  " Open file totally unfolded

call plug#begin('/opt/nvim-plugged')
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
Plug 'dense-analysis/ale'
Plug 'davidhalter/jedi-vim'
Plug 'Shougo/echodoc.vim'
Plug 'preservim/tagbar'
Plug 'pixelneo/vim-python-docstring'
Plug 'pseewald/vim-anyfold'
Plug 'christoomey/vim-tmux-navigator'
Plug 'airblade/vim-gitgutter'
call plug#end()

let g:airline_theme='papercolor'
let g:airline#extensions#tagbar#enabled = 1

" Only run linters named in ale_linters settings.
let g:ale_linters_explicit = 1
let g:ale_linters = {'python': ['flake8']}
let g:ale_fixers = {'python': ['yapf']}
let g:ale_fix_on_save = 1
let g:ale_lint_on_text_changed = 1
let g:ale_sign_priority = 0

let g:jedi#popup_on_dot = 0
let g:jedi#use_splits_not_buffers = 'winwidth'
let g:jedi#show_call_signatures = "0"

set cmdheight=2
let g:echodoc#enable_at_startup = 1

"tagbar configuration
noremap tt :TagbarToggle<CR>
let g:tagbar_sort = 0  " Do not sort tags
noremap t :TagbarOpen j<CR>

autocmd Filetype * AnyFoldActivate
let g:anyfold_fold_comments=1

let g:gitgutter_max_signs=10000
let g:gitgutter_override_sign_column_highlight = 0
let g:gitgutter_preview_win_floating = 0
"let g:gitgutter_highlight_lines = 1
" No line highlight for deletions  
" highlight GitGutterDeleteLine ctermfg=0 ctermbg=0
" Use fontawesome icons as signs
let g:gitgutter_sign_added = '+'
let g:gitgutter_sign_modified = '>'
let g:gitgutter_sign_removed = '-'
let g:gitgutter_sign_removed_first_line = '^'
let g:gitgutter_sign_modified_removed = '<'

let g:gitgutter_sign_priority = 0
let g:gitgutter_sign_allow_clobber = 0
" colors for signs:
highlight GitGutterAdd ctermbg=Green
highlight GitGutterDelete ctermbg=Red
highlight GitGutterChange ctermbg=Blue
" Key map for hunks
nmap ghs <Plug>(GitGutterStageHunk)
nmap ghu <Plug>(GitGutterUndoHunk)
nmap ghp <Plug>(GitGutterPreviewHunk)
" Update sign column every quarter second
set updatetime=250

let g:tmux_navigator_no_mappings = 1
nnoremap <silent> <S-Left> :TmuxNavigateLeft<cr>
nnoremap <silent> <S-Down> :TmuxNavigateDown<cr>
nnoremap <silent> <S-Up> :TmuxNavigateUp<cr>
nnoremap <silent> <S-Right> :TmuxNavigateRight<cr>

autocmd BufNewFile *.sh 0r ~/config/sh_template_file.sh
autocmd BufNewFile *.py 0r ~/config/py_template_file.py
