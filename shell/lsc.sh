#!/usr/bin/env bash
# lsc: List gnome ShortCuts

gsettings list-recursively org.gnome.desktop.wm.keybindings | fzf
