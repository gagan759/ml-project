"""Page registry, so pages can link to each other (filled by streamlit_app.py before the page runs)."""
import streamlit as st

PAGES: dict = {}


def go(key: str) -> None:
    st.switch_page(PAGES[key])


def link(key: str, label: str, icon: str | None = None) -> None:
    st.page_link(PAGES[key], label=label, icon=icon)
