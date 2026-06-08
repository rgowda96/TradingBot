#!/usr/bin/env python3
import functools
import http.server
import socketserver

DIRECTORY = "/Users/rakshaksgowda/Desktop/shayan's custom strategy"
PORT = 8765

Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=DIRECTORY)
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    print(f"serving {DIRECTORY} on http://127.0.0.1:{PORT}")
    httpd.serve_forever()
