# Streamlit.py  --  Minimal Streamlit UI

# Just a text box, a submit button, and the response.
# It calls the FastAPI backend with requests.post().


import requests
import streamlit as st

API_URL = "http://localhost:8000/ask"

st.title("Enterprise Customer Support Assistant")
st.write("Ask a question and the multi-agent RAG system will answer.")

# Text input
user_query = st.text_input("Your question:")

# Submit button
if st.button("Ask"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running multi-agent workflow..."):
            try:
                # Call the FastAPI /ask endpoint
                resp = requests.post(API_URL, json={"query": user_query}, timeout=120)
                data = resp.json()

                # Response
                st.subheader("Response")
                st.write(data.get("response", "No response"))

                # Escalation flag
                if data.get("escalate_flag"):
                    st.error("This issue has been flagged for human escalation.")
                else:
                    st.success("Handled by AI — no escalation needed.")

                # Show sentiment line (raw)
                st.caption(data.get("sentiment", ""))

                # Retrieved documents
                st.subheader("Retrieved Documents")
                for i, src in enumerate(data.get("sources", []), 1):
                    with st.expander(f"Source {i} — {src['metadata'].get('category', 'general')}"):
                        st.write(src["content"])

            except Exception as e:
                st.error(f"Request failed: {e}")
