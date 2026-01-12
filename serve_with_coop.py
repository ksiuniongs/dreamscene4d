import http.server
import socketserver


class COOPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    with socketserver.TCPServer(("", 8008), COOPHandler) as httpd:
        print("Serving with COOP/COEP at http://localhost:8008")
        httpd.serve_forever()
