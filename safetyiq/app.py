"""
SafetyIQ Streamlit UI

This is the front-end of SafetyIQ. It imports the RAG chain from src/rag_chain.py and wraps
it up in a clean web interface.
"""

import sys
import os

import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rag_chain import load_vectorstore, build_rag_chain, query



# Page configuration
st.set_page_config(
    page_title="SafetyIQ",
    page_icon="🛡️",
    layout="centered"
)

# Load the RAG chain (cached)
@st.cache_resource
def load_chain():
    """Load vectorstore and build RAG chain.
    Runs once, then cached for the lifetime of the app session"""
    vectorstore = load_vectorstore()
    rag_chain,retriever = build_rag_chain(vectorstore)
    return rag_chain, retriever


# === Header ===
st.title("🛡️ SafetyIQ")
st.markdown("**Industrial Safety & Equipment Documentation Assistant**")
st.markdown(
    "Ask questions about OSHA standards, process safety management, "
    "equipment procedures, and fire alarm systems. "
    "Answers are sourced directly from the loaded documentation."
)

st.divider()


# === Query input ===
question = st.text_area(
    "Ask a question:",
    placeholder=(
        "e.g. What are the key elements of a Process Safety Management program?\n"
        "e.g. What PPE is required when working with corrosive chemicals?\n"
        "e.g. What are the inspection intervals for control valves?"

    ),
    height=120
)

ask_button = st.button("Ask SafetyIQ", use_container_width=True, type="primary")


# === Answer + Sources ===
if ask_button:
    if not question.strip():
        st.warning("PLease enter a question before clicking Ask.")
    else:
        rag_chain, retriever = load_chain()

        with st.spinner("Searching documentation and generating answer..."):
            result = query(question.strip(), rag_chain, retriever)

        
        # === Display the Answer ===
        st.markdown("### Answer")

        no_answer_signal = "couldn't find this in the loaded documents"
        if no_answer_signal in result["answer"].lower():
            st.info(result["answer"])
        else:
            st.success(result["answer"])

        # === Display the sources ===
        st.markdown("### Sources")
        st.caption("The following document sections were retrieved to answer your question:")

        if result["sources"]:
            num_cols = min(len(result["sources"]), 3)
            cols = st.columns(num_cols)
 
            for i, source in enumerate(result["sources"]):
                with cols[i % num_cols]:
    
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #1e2a3a;
                            border: 1px solid #2d4a6e;
                            border-radius: 8px;
                            padding: 12px 16px;
                            margin-bottom: 8px;
                        ">
                            <div style="font-size: 0.75rem; color: #7aa0c4; margin-bottom: 4px;">
                                📄 SOURCE
                            </div>
                            <div style="font-size: 0.85rem; font-weight: 600; color: #e8edf2;">
                                {source['file']}
                            </div>
                            <div style="font-size: 0.8rem; color: #a0b4c8; margin-top: 4px;">
                                Page {source['page']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.caption("No sources could be identified.")


# === Footer ===
st.divider()
st.caption(
    "SafetyIQ answers questions based on loaded documentation only. "
    "Always verify critical safety information against official sources."
)
