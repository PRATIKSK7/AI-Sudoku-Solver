import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_frontend_imports():
    """
    Test that the frontend app can be imported without syntax or initialization errors.
    Since streamlit runs in a specific context, we just verify the file parses successfully.
    """
    try:
        import frontend.app
        success = True
    except Exception as e:
        success = False
        
    assert success == True
