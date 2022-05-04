import os
import sys
from configure_logging import configure_logging

from services import create_app

app = create_app(os.getenv("FLASK_ENV") or "test")
if __name__ == "__main__":
    configure_logging()

    port = 5555
    if len(sys.argv) > 1:
        port = sys.argv[1]
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)