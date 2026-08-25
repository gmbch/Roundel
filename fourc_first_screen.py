"""Standalone entry point for testing the reusable Roundel 4C viewer."""

import os

import streamlit as st

from fourc_viewer import render_4c_review


DEFAULT_STUDY_UID = "1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0"

st.set_page_config(page_title="Roundel 4C First Screen", page_icon="🫀", layout="wide")
st.title("Roundel 4C Candidate Review")
st.caption("Standalone viewer for validating mind-map and Ambra access before production use.")

with st.sidebar:
    study_uid = st.text_input("Study UID", value=DEFAULT_STUDY_UID).strip()
    site_code = st.text_input("Site", value=os.getenv("ROUNDL_DEFAULT_SITE", "CIN")).strip().upper()

if study_uid and site_code:
    render_4c_review(study_uid, site_code, {"site": site_code})
else:
    st.info("Enter a study UID and site to load 4C candidates.")
