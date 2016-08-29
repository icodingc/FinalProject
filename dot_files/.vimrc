set nocompatible
set nobackup
set noswapfile

set t_Co=256

filetype off
filetype plugin indent on

let mapleader=";"
nmap <leader>w <C-W>
set pastetoggle=<leader>pp

set incsearch
set ignorecase
set backspace=indent,eol,start

set wrap
set wildmenu
set wildmode=list:full
set completeopt=longest,menu
set wildignore+=*.a,*.o
set wildignore+=*.bmp,*.gif,*.ico,*.jpg,*.png
set wildignore+=.DS_Store,.git,.hg,.svn
set wildignore+=*~,*.swp,*.tmp

set mouse=a
autocmd BufEnter * let &titlestring = ' ' . expand("%:t")
set title

set laststatus=2
set ruler
set number

set hlsearch

syntax enable
syntax on

filetype indent on
set expandtab

" 默认缩进4
set ai
set sw=4
set ts=4
set sts=4

set mousemodel=popup

map j gj
map k gk

" insert data, time
nmap <Leader>it a<C-R>=strftime("%H:%M")<CR><Esc>
nmap <Leader>id a<C-R>=strftime("%Y-%m-%d")<CR><Esc>
nmap <Leader>t i<C-R>=strftime("%Y-%m-%d %H:%M")<CR><Esc>
let g:plug_url_format = 'git@github.com:%s.git'
call plug#begin('~/.vim/plugged')
"Plug 'Shougo/neocomplete.vim'
Plug 'Shougo/neocomplcache.vim'
Plug 'davidhalter/jedi-vim'
Plug 'SirVer/ultisnips'
Plug 'honza/vim-snippets'
Plug 'Yggdroot/indentLine'
Plug 'itchyny/lightline.vim'
Plug 'junegunn/vim-easy-align'
Plug 'scrooloose/nerdcommenter'
Plug 'jiangmiao/auto-pairs'
Plug 'scrooloose/nerdtree'
Plug 'junegunn/fzf', { 'dir': '~/.fzf', 'do': './install --all'  }
Plug 'junegunn/fzf.vim'
Plug 'NLKNguyen/papercolor-theme'
Plug 'scrooloose/syntastic'
Plug 'mbbill/undotree'
call plug#end()

" indentLine
let g:indentLine_color_term = 9

nmap <Leader>lf :NERDTreeToggle<CR>
" easytags
let g:easytags_events = ['BufWritePost']
let g:easytags_file = '~/.vim/tags'
nmap <Leader>lt :TagbarToggle<CR>

colo PaperColor
set bg=dark

let g:UltiSnipsExpandTrigger="<leader><tab>"
let g:UltiSnipsJumpForwardTrigger="<leader><tab>"
let g:UltiSnipsJumpBackwardTrigger="<leader><s-tab>"


" fzf
set wildignore+=*/tmp/*,*.so,*.swp,*.zip,*.pyc,*.pickle     " MacOSX/Linux
nmap <leader>ff :FZF<CR>
nmap <leader>fb :Buffers<CR>
nmap <leader>ft :BTags<CR>
nmap <leader>fl :BLines<CR>
nmap <leader>fh :History<CR>


" vim-ag-anying.vim ag anything
" Maintainer:   Chun Yang <http://github.com/Chun-Yang>
" Version:      1.0
" 使用fzf.vim的Ag命令替代ag.vim

" http://stackoverflow.com/questions/399078/what-special-characters-must-be-escaped-in-regular-expressions
let g:vim_action_ag_escape_chars = get(g:, 'vim_action_ag_escape_chars', '#%.^$*+?()[{\\|')

function! s:Ag(mode) abort
  " preserver @@ register
  let reg_save = @@

  " copy selected text to @@ register
  if a:mode ==# 'v' || a:mode ==# ''
    silent exe "normal! `<v`>y"
  elseif a:mode ==# 'char'
    silent exe "normal! `[v`]y"
  else
    return
  endif

  " escape special chars,
  " % is file name in vim we need to escape that first
  " # is secial in ag
  let escaped_for_ag = escape(@@, '%#')
  let escaped_for_ag = escape(escaped_for_ag, g:vim_action_ag_escape_chars)

  " execute Ag command
  " '!' is used to NOT jump to the first match
  exe ":Ag ".escaped_for_ag

  " recover @@ register
  let @@ = reg_save
endfunction

vnoremap <silent> <Plug>AgActionVisual :<C-U>call <SID>Ag(visualmode())<CR>
vmap <leader><space> <Plug>AgActionVisual

" syntastic

let g:syntastic_always_populate_loc_list = 0
let g:syntastic_auto_loc_list = 0
let g:syntastic_check_on_open = 1
let g:syntastic_check_on_wq = 0

if has("persistent_undo")
    set undodir=~/.undodir/
    set undofile
endif
nmap <leader>lu :UndotreeToggle<cr>

" Disable AutoComplPop.
let g:acp_enableAtStartup = 0
" Use neocomplete.
let g:neocomplete#enable_at_startup = 1
" Use smartcase.
let g:neocomplete#enable_smart_case = 1
" Set minimum syntax keyword length.
let g:neocomplete#sources#syntax#min_keyword_length = 1
let g:neocomplete#lock_buffer_name_pattern = '\*ku\*'

" Define dictionary.
let g:neocomplete#sources#dictionary#dictionaries = {
    \ 'default' : ''
        \ }

" Define keyword.
if !exists('g:neocomplete#keyword_patterns')
    let g:neocomplete#keyword_patterns = {}
endif
let g:neocomplete#keyword_patterns['default'] = '\h\w*'

" Plugin key-mappings.
inoremap <expr><C-g>     neocomplete#undo_completion()
inoremap <expr><C-l>     neocomplete#complete_common_string()

" Recommended key-mappings.
" <CR>: close popup and save indent.
inoremap <silent> <CR> <C-r>=<SID>my_cr_function()<CR>
function! s:my_cr_function()
  return (pumvisible() ? "\<C-y>" : "" ) . "\<CR>"
  " For no inserting <CR> key.
  "return pumvisible() ? "\<C-y>" : "\<CR>"
endfunction
" <TAB>: completion.
inoremap <expr><TAB>  pumvisible() ? "\<C-n>" : "\<TAB>"
" <C-h>, <BS>: close popup and delete backword char.
inoremap <expr><C-h> neocomplete#smart_close_popup()."\<C-h>"
inoremap <expr><BS> neocomplete#smart_close_popup()."\<C-h>"
" Close popup by <Space>.
"inoremap <expr><Space> pumvisible() ? "\<C-y>" : "\<Space>"

" Enable omni completion.
autocmd FileType css setlocal omnifunc=csscomplete#CompleteCSS
autocmd FileType html,markdown setlocal omnifunc=htmlcomplete#CompleteTags
autocmd FileType javascript setlocal omnifunc=javascriptcomplete#CompleteJS
autocmd FileType python setlocal omnifunc=jedi#completions
autocmd FileType xml setlocal omnifunc=xmlcomplete#CompleteTags

" Enable heavy omni completion.
if !exists('g:neocomplete#sources#omni#input_patterns')
  let g:neocomplete#sources#omni#input_patterns = {}
endif

" jedivim
let g:jedi#goto_command = "<leader>jg"
let g:jedi#goto_assignments_command = "<leader>ja"
let g:jedi#goto_definitions_command = "<leader>jd"
let g:jedi#documentation_command = "<leader>jK"
let g:jedi#usages_command = "<leader>ju"
let g:jedi#completions_command = ""
let g:jedi#rename_command = "<leader>jr"
