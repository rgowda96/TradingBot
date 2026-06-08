#!/usr/bin/env python3
import os, sys
os.chdir(os.path.join(os.path.dirname(__file__), "dashboard"))
from http.server import HTTPServer, SimpleHTTPRequestHandler
HTTPServer(("", 7788), SimpleHTTPRequestHandler).serve_forever()
