import http.server
import socketserver
def main(in_q):
    
    print('placeholder')
    PORT = 8000
    DIRECTORY = "visualizer"
    #DIRECTORY = "exhibit\visualization"


    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {
            '': 'application/octet-stream',
            '.manifest': 'text/cache-manifest',
            '.html': 'text/html',
            '.css':	'text/css',
            '.js':'text/javascript',
            '.wasm': 'application/wasm',
            '.json': 'application/json',
            '.xml': 'application/xml',
        }
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)


    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()

if __name__ == "__main__":
    main("")
