# streamlit_test.py
# Run with: streamlit run streamlit_test.py

import streamlit as st

# ── TEXT ELEMENTS ──────────────────────────────────────
st.title("This is a big title")
st.header("This is a header")
st.subheader("This is a subheader")
st.write("This is normal text — write() is the most versatile command")
st.markdown("**Bold**, *italic*, `code` — markdown works here")

# ── INPUT ELEMENTS ─────────────────────────────────────
name = st.text_input("Type your name here:")
st.write(f"You typed: {name}")

age = st.slider("Select your age:", min_value=18, max_value=60, value=25)
st.write(f"Age selected: {age}")

text_block = st.text_area("Paste a long text here:", height=150)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file is not None:
    st.write(f"File uploaded: {uploaded_file.name}")

# ── BUTTONS ────────────────────────────────────────────
if st.button("Click me!"):
    st.write("Button was clicked!")
    # Everything inside this if block runs ONLY when button is clicked

# ── LAYOUT ─────────────────────────────────────────────
col1, col2 = st.columns(2)       # splits page into 2 equal columns
with col1:
    st.write("Left column content")
with col2:
    st.write("Right column content")

# ── VISUAL FEEDBACK ────────────────────────────────────
st.success("This is a green success message")
st.warning("This is a yellow warning message")
st.error("This is a red error message")
st.info("This is a blue info message")

# ── PROGRESS & LOADING ─────────────────────────────────
with st.spinner("Loading..."):
    import time
    time.sleep(2)           # simulate work being done
st.success("Done!")

# ── METRICS (big number displays) ─────────────────────
st.metric(label="Match Score", value="78%", delta="+12% vs last resume")