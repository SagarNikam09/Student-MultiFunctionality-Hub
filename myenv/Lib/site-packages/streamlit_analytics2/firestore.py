import json

import streamlit as st
from google.cloud import firestore
from google.oauth2 import service_account


def sanitize_data(data):
    if isinstance(data, dict):
        # Recursively sanitize dictionary keys
        return {str(k) if k else "": sanitize_data(v) for k, v in data.items() if k}
    elif isinstance(data, list):
        # Apply sanitization to elements in lists
        return [sanitize_data(item) for item in data]
    else:
        return data


def load(
    counts,
    service_account_json,
    collection_name,
    streamlit_secrets_firestore_key,
    firestore_project_name,
):
    """Load count data from firestore into `counts`."""
    if streamlit_secrets_firestore_key is not None:
        # Following along here
        # https://blog.streamlit.io/streamlit-firestore-continued/#part-4-securely-deploying-on-streamlit-sharing for deploying to Streamlit Cloud with Firestore
        key_dict = json.loads(st.secrets[streamlit_secrets_firestore_key])
        creds = service_account.Credentials.from_service_account_info(key_dict)
        db = firestore.Client(credentials=creds, project=firestore_project_name)
        col = db.collection(collection_name)
        firestore_counts = col.document("counts").get().to_dict()
    else:
        db = firestore.Client.from_service_account_json(service_account_json)
        col = db.collection(collection_name)
        firestore_counts = col.document("counts").get().to_dict()

    if firestore_counts is not None:
        for key in firestore_counts:
            if key in counts:
                counts[key] = firestore_counts[key]

    # Log loaded data for debugging
    # logging.debug("Data loaded from Firestore: %s", firestore_counts)


def save(
    counts,
    service_account_json,
    collection_name,
    streamlit_secrets_firestore_key,
    firestore_project_name,
):
    """Save count data from `counts` to firestore."""
    # Ensure all keys are strings and not empty
    sanitized_counts = sanitize_data(counts)

    if streamlit_secrets_firestore_key is not None:
        # Following along here https://blog.streamlit.io/streamlit-firestore-continued/#part-4-securely-deploying-on-streamlit-sharing for deploying to Streamlit Cloud with Firestore
        key_dict = json.loads(st.secrets[streamlit_secrets_firestore_key])
        creds = service_account.Credentials.from_service_account_info(key_dict)
        db = firestore.Client(credentials=creds, project=firestore_project_name)
    else:
        db = firestore.Client.from_service_account_json(service_account_json)
    col = db.collection(collection_name)
    doc = col.document("counts")
    # TODO pass user set argument via config screen for the name of document
    # currently hard coded to be "counts"

    # Log the data being saved
    # logging.debug("Data being saved to Firestore: %s", sanitized_counts)

    # Attempt to save to Firestore
    doc.set(sanitized_counts)  # creates if doesn't exist
