# Code Completion for EveryVoice config files

When manually editing EveryVoice's configuration files, it is convenient to have the file checked/validated and to have documentation about each field.
Each EveryVoice config file format is defined by a schema that enables code completion and access to documentation right in the editor, for both json and yaml configuration files.

## Code completion in VSCode and other IDEs

We publish the code completion schemas to https://www.schemastore.org/ and VSCode automatically uses these, so it should just work.
The same will be true of several other IDEs.

## How to Setup Code Completion for Schemas in vim

In Vim, it's not automatic, but the following recipe will let you enable code completion.

### Install `nodejs`

You will need to have a functional `npm` which is part of `nodejs`.
The schemas will be verified using a node process.

### Install vim-plug

[vim-plug](https://github.com/junegunn/vim-plug):  Minimalist Vim Plugin Manager
This will take care of install vim's extensions for us.

```sh
curl \
  --create-dirs \
  -fLo ~/.vim/autoload/plug.vim \
  https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

### Augment your `.vimrc`

We want to install [Conquer of Completion aka coc](https://github.com/neoclide/coc.nvim)
We will add a plugins for [coc-json](https://github.com/neoclide/coc-json) and [coc-yaml](https://github.com/neoclide/coc-yaml) which will be used to handle json and yaml files.
Let's also add some key bindings to access `coc.nvim`'s functionalities.
Refer to [Example Vim configuration](https://github.com/neoclide/coc.nvim#example-vim-configuration) for a more complete example.

```
call plug#begin()
Plug 'neoclide/coc.nvim', { 'do': 'npm ci' }

let g:coc_disable_startup_warning = 1

" Use tab for trigger completion with characters ahead and navigate
" NOTE: There's always complete item selected by default, you may want to enable
" no select by `"suggest.noselect": true` in your configuration file
" NOTE: Use command ':verbose imap <tab>' to make sure tab is not mapped by
" other plugin before putting this into your config
inoremap <silent><expr> <TAB>
\ coc#pum#visible() ? coc#pum#next(1) :
\ CheckBackspace() ? "\<Tab>" :
\ coc#refresh()
inoremap <expr><S-TAB> coc#pum#visible() ? coc#pum#prev(1) : "\<C-h>"

" Make <CR> to accept selected completion item or notify coc.nvim to format
" <C-g>u breaks current undo, please make your own choice
inoremap <silent><expr> <CR> coc#pum#visible() ? coc#pum#confirm()
\: "\<C-g>u\<CR>\<c-r>=coc#on_enter()\<CR>"

function! CheckBackspace() abort
  let col = col('.') - 1
  return !col || getline('.')[col - 1]  =~# '\s'
endfunction

" Use <c-space> to trigger completion
if has('nvim')
  inoremap <silent><expr> <c-space> coc#refresh()
else
  inoremap <silent><expr> <c-@> coc#refresh()
endif

" Use `[g` and `]g` to navigate diagnostics
" Use `:CocDiagnostics` to get all diagnostics of current buffer in location list
nmap <silent> [g <Plug>(coc-diagnostic-prev)
nmap <silent> ]g <Plug>(coc-diagnostic-next)

" GoTo code navigation
nmap <silent> gd <Plug>(coc-definition)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gr <Plug>(coc-references)

" Use K to show documentation in preview window
nnoremap <silent> K :call ShowDocumentation()<CR>

function! ShowDocumentation()
  if CocAction('hasProvider', 'hover')
    call CocActionAsync('doHover')
  else
    call feedkeys('K', 'in')
  endif
endfunction

" Highlight the symbol and its references when holding the cursor
autocmd CursorHold * silent call CocActionAsync('highlight')

" Symbol renaming
nmap <leader>rn <Plug>(coc-rename)

" Formatting selected code
xmap <leader>F  <Plug>(coc-format-selected)
nmap <leader>F  <Plug>(coc-format-selected)

augroup mygroup
autocmd!
" Setup formatexpr specified filetype(s)
autocmd FileType typescript,json setl formatexpr=CocAction('formatSelected')
" Update signature help on jump placeholder
autocmd User CocJumpPlaceholder call CocActionAsync('showSignatureHelp')
augroup end

" Applying code actions to the selected code block
" Example: `<leader>aap` for current paragraph
xmap <leader>a  <Plug>(coc-codeaction-selected)
nmap <leader>a  <Plug>(coc-codeaction-selected)

" Remap keys for applying code actions at the cursor position
nmap <leader>ac  <Plug>(coc-codeaction-cursor)
" Remap keys for apply code actions affect whole buffer
nmap <leader>as  <Plug>(coc-codeaction-source)
" Apply the most preferred quickfix action to fix diagnostic on the current line
nmap <leader>qf  <Plug>(coc-fix-current)

" Remap keys for applying refactor code actions
nmap <silent> <leader>re <Plug>(coc-codeaction-refactor)
xmap <silent> <leader>r  <Plug>(coc-codeaction-refactor-selected)
nmap <silent> <leader>r  <Plug>(coc-codeaction-refactor-selected)

" Run the Code Lens action on the current line
nmap <leader>cl  <Plug>(coc-codelens-action)

" Map function and class text objects
" NOTE: Requires 'textDocument.documentSymbol' support from the language server
xmap if <Plug>(coc-funcobj-i)
omap if <Plug>(coc-funcobj-i)
xmap af <Plug>(coc-funcobj-a)
omap af <Plug>(coc-funcobj-a)
xmap ic <Plug>(coc-classobj-i)
omap ic <Plug>(coc-classobj-i)
xmap ac <Plug>(coc-classobj-a)
omap ac <Plug>(coc-classobj-a)

" Remap <C-f> and <C-b> to scroll float windows/popups
if has('nvim-0.4.0') || has('patch-8.2.0750')
nnoremap <silent><nowait><expr> <C-f> coc#float#has_scroll() ? coc#float#scroll(1) : "\<C-f>"
nnoremap <silent><nowait><expr> <C-b> coc#float#has_scroll() ? coc#float#scroll(0) : "\<C-b>"
inoremap <silent><nowait><expr> <C-f> coc#float#has_scroll() ? "\<c-r>=coc#float#scroll(1)\<cr>" : "\<Right>"
inoremap <silent><nowait><expr> <C-b> coc#float#has_scroll() ? "\<c-r>=coc#float#scroll(0)\<cr>" : "\<Left>"
vnoremap <silent><nowait><expr> <C-f> coc#float#has_scroll() ? coc#float#scroll(1) : "\<C-f>"
vnoremap <silent><nowait><expr> <C-b> coc#float#has_scroll() ? coc#float#scroll(0) : "\<C-b>"
endif

" Use CTRL-S for selections ranges
" Requires 'textDocument/selectionRange' support of language server
nmap <silent> <C-s> <Plug>(coc-range-select)
xmap <silent> <C-s> <Plug>(coc-range-select)

" Add `:Format` command to format current buffer
command! -nargs=0 Format :call CocActionAsync('format')

" Add `:Fold` command to fold current buffer
command! -nargs=? Fold :call     CocAction('fold', <f-args>)

" Add `:OR` command for organize imports of the current buffer
command! -nargs=0 OR   :call     CocActionAsync('runCommand', 'editor.action.organizeImport')

" Add (Neo)Vim's native statusline support
" NOTE: Please see `:h coc-status` for integrations with external plugins that
" provide custom statusline: lightline.vim, vim-airline
set statusline^=%{coc#status()}%{get(b:,'coc_current_function','')}

" TODO Space is our Leader, this might interfer with the following:
" Mappings for CoCList
" Show all diagnostics
nnoremap <silent><nowait> <space>a  :<C-u>CocList diagnostics<cr>
" Manage extensions
nnoremap <silent><nowait> <space>e  :<C-u>CocList extensions<cr>
" Show commands
nnoremap <silent><nowait> <space>c  :<C-u>CocList commands<cr>
" Find symbol of current document
nnoremap <silent><nowait> <space>o  :<C-u>CocList outline<cr>
" Search workspace symbols
nnoremap <silent><nowait> <space>s  :<C-u>CocList -I symbols<cr>
" Do default action for next item
nnoremap <silent><nowait> <space>j  :<C-u>CocNext<CR>
" Do default action for previous item
nnoremap <silent><nowait> <space>k  :<C-u>CocPrev<CR>
" Resume latest coc list
nnoremap <silent><nowait> <space>p  :<C-u>CocListResume<CR>
call plug#end()
```

### Install the New Plugins

Plugins don't automatically install themself thus you have to run the following command to install them.
Start `vim` then do

```sh
vim +PlugInstall "+:CocInstall coc-json" "+:CocInstall coc-yaml" +:qall
```

### Compile coc.nvim

Once your plugins are installed, you will need to compile coc.

```
cd ~/.vim/plugged/coc.nvim
npm ci
```

### Create Coc-settings.json

Start `vim` and run the command `:CocConfig` to edit where your everyvoice schemas are located.
The following example assumes that you have clone EveryVoice into `~/git/EveryVoice`.
Make the proper modifications to match where you have cloned EveryVoice.
Also note that you have to change `/home/username` with your own username in the yaml section.

```json
{
  "json.schemas": [
    {
      "url": "file://${userHome}/git/EveryVoice/everyvoice/.schema/everyvoice-shared-data-0.1.json",
      "fileMatch": [
        "everyvoice-shared-data.json"
      ]
    },
    {
      "url": "file://${userHome}/git/EveryVoice/everyvoice/.schema/everyvoice-shared-text-0.1.json",
      "fileMatch": [
        "everyvoice-shared-text.json"
      ]
    },
    {
      "url": "file://${userHome}/git/EveryVoice/everyvoice/.schema/everyvoice-spec-to-wav-0.1.json",
      "fileMatch": [
        "everyvoice-spec-to-wav.json"
      ]
    },
    {
      "url": "file://${userHome}/git/EveryVoice/everyvoice/.schema/everyvoice-text-to-spec-0.1.json",
      "fileMatch": [
        "everyvoice-text-to-spec.json"
      ]
    },
    {
      "url": "file://${userHome}/git/EveryVoice/everyvoice/.schema/everyvoice-text-to-wav-0.1.json",
      "fileMatch": [
        "everyvoice-text-to-wav.json"
      ]
    }
  ],
  "yaml.schemas": {
    "file://home/username/git/EveryVoice/everyvoice/.schema/everyvoice-shared-data-0.1.json": [
      "everyvoice-shared-data.yaml"
    ],
    "file://home/username/git/EveryVoice/everyvoice/.schema/everyvoice-shared-text-0.1.json": [
      "everyvoice-shared-text.yaml"
    ],
    "file://home/username/git/EveryVoice/everyvoice/.schema/everyvoice-spec-to-wav-0.1.json": [
      "everyvoice-spec-to-wav.yaml"
    ],
    "file://home/username/git/EveryVoice/everyvoice/.schema/everyvoice-text-to-spec-0.1.json": [
      "everyvoice-text-to-spec.yaml"
    ],
    "file://home/username/git/EveryVoice/everyvoice/.schema/everyvoice-text-to-wav-0.1.json": [
      "everyvoice-text-to-wav.yaml"
    ]
  }
}
```

### Usage

Once everything is installed, start editing a new or existing EveryVoice configuration.

```sh
vim everyvoice-shared-data.json
```

Then use `CTRL+<space>` to trigger completion.
