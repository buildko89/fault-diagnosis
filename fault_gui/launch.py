"""Console-script launcher: ``fault-gui`` -> ``streamlit run app.py``."""
import os
import sys


def main():
    from streamlit.web import cli as stcli

    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
